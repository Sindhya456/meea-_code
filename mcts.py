#!/usr/bin/env python3
"""
Complete MCTS + UQ Pipeline for Lambda Labs
Runs: Phase 1 (generate) → Phase 2 (UQ) → Phase 3 (MCTS/A* with UQ)

Usage:
  python run_pipeline.py --dataset USPTO --gpu 0 --phase all
  python run_pipeline.py --dataset USPTO --gpu 0 --phase 1  # Just generate
  python run_pipeline.py --dataset USPTO --gpu 0 --phase 2  # Just UQ
  python run_pipeline.py --dataset USPTO --gpu 0 --phase 3  # Just search
"""

import pickle, torch, torch.nn.functional as F, numpy as np, pandas as pd
import os, sys, argparse, time, signal, heapq
from contextlib import contextmanager

# Your imports - make sure these files are in same directory
from valueEnsemble import ValueEnsemble
from policyNet import MLPModel
from rdkit import Chem
from rdkit.Chem import AllChem

print("="*60)
print("MCTS + UQ Pipeline for Lambda Labs")
print("="*60)

# ============================================================
# UTILITIES (Phase 0)
# ============================================================

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

class SmilesEnumerator:
    def __init__(self, enum=True, canonical=False, isomericSmiles=True):
        self.enumerate = enum
        self.canonical = canonical
        self.isomericSmiles = isomericSmiles
    
    def randomize_smiles(self, smiles):
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return smiles
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        return Chem.MolToSmiles(nm, canonical=self.canonical, 
                               isomericSmiles=self.isomericSmiles)

def smiles_to_fp(s, fp_dim=2048):
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return np.zeros(fp_dim, dtype=np.bool_)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool_)
    arr[list(fp.GetOnBits())] = 1
    return arr

def batch_smiles_to_fp(s_list, fp_dim=2048):
    return np.array([smiles_to_fp(s, fp_dim) for s in s_list])

def value_fn(model, mols, device):
    num_mols = len(mols)
    fps = batch_smiles_to_fp(mols).reshape(num_mols, -1)
    if len(fps) <= 5:
        mask = np.ones(5); mask[len(fps):] = 0
        fps_input = np.zeros((5, 2048)); fps_input[:len(fps)] = fps
    else:
        mask = np.ones(len(fps)); fps_input = fps
    fps_t = torch.FloatTensor([fps_input]).to(device)
    mask_t = torch.FloatTensor([mask]).to(device)
    return model(fps_t, mask_t).cpu().data.numpy()[0][0]

def prepare_value(model_f, gpu):
    device = 'cpu' if gpu == -1 else f'cuda:{gpu}'
    model = ValueEnsemble(2048, 128, 0.1).to(device)
    model.load_state_dict(torch.load(model_f, map_location=device))
    model.eval()
    return model

def prepare_expand(model_path, gpu):
    device = 'cpu' if gpu == -1 else f'cuda:{gpu}'
    return MLPModel(model_path, './saved_model/template_rules.dat', device=device)

def prepare_starting_molecules():
    path = './prepare_data/origin_dict.csv'
    if not os.path.exists(path):
        print(f"[WARN] {path} not found")
        return set()
    return set(pd.read_csv(path)['mol'].tolist())

# ============================================================
# PHASE 1: GENERATE POLICY LOGITS & VALUE PREDS
# ============================================================

def generate_policy_logits(expand_fn, mols, out_path, n_aug=20, topk=50):
    """Generate policy logits with TTA"""
    print(f"\n[PHASE 1a] Generating policy logits...")
    print(f"  Molecules: {len(mols)}, Augmentations: {n_aug}")
    
    sm_en = SmilesEnumerator(enum=True, canonical=False)
    results = []
    
    with torch.no_grad():
        for idx, s in enumerate(mols):
            aug_candidates, aug_scores = [], []
            for _ in range(n_aug):
                try:
                    aug_s = sm_en.randomize_smiles(s)
                except:
                    aug_s = s
                out = expand_fn.run(aug_s, topk=topk)
                if out is None:
                    aug_candidates.append([]); aug_scores.append(np.array([]))
                else:
                    aug_candidates.append(list(out['reactants']))
                    aug_scores.append(np.array(out['scores'], dtype=np.float32))
            
            union = []
            for cands in aug_candidates:
                for c in cands:
                    if c not in union: union.append(c)
            if not union: continue
            
            aligned = np.zeros((n_aug, len(union)), dtype=np.float32)
            for i, (cands, scores) in enumerate(zip(aug_candidates, aug_scores)):
                for j, c in enumerate(cands):
                    aligned[i, union.index(c)] = scores[j]
            
            agg_scores = aligned.mean(axis=0)
            results.append({"mol": s, "candidates": union, "agg_scores": agg_scores})
            
            if (idx+1) % 10 == 0:
                print(f"  {idx+1}/{len(mols)} ({100*(idx+1)/len(mols):.1f}%)")
    
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, np.array(results, dtype=object))
    print(f"✅ Saved to {out_path}")

def generate_value_preds(value_model, mols, device, out_path):
    """Generate value predictions"""
    print(f"\n[PHASE 1b] Generating value predictions...")
    preds = []
    for idx, s in enumerate(mols):
        preds.append(value_fn(value_model, [s], device))
        if (idx+1) % 10 == 0:
            print(f"  {idx+1}/{len(mols)} ({100*(idx+1)/len(mols):.1f}%)")
    pd.DataFrame({"mol": mols, "value": preds}).to_csv(out_path, index=False)
    print(f"✅ Saved to {out_path}")

# ============================================================
# PHASE 2: GENERATE UQ CSV FILES
# ============================================================

def compute_uq(policy_logits, value_preds, alea_w=0.5, epis_w=0.5):
    """Compute combined UQ scores"""
    B, C = policy_logits.shape
    
    # Aleatoric (entropy)
    aleatoric = []
    for i in range(B):
        probs = F.softmax(policy_logits[i].unsqueeze(0), dim=1)
        mean_probs = probs.mean(dim=0)
        entropy = -(mean_probs * torch.log(mean_probs + 1e-12)).sum().item()
        aleatoric.append(entropy)
    aleatoric = np.array(aleatoric)
    
    # Epistemic (JSD)
    policy_probs = F.softmax(policy_logits, dim=1)
    value_probs = torch.sigmoid(value_preds).squeeze(-1)
    value_probs = torch.stack([value_probs, 1-value_probs], dim=1)
    if policy_probs.shape[1] != 2:
        top2, _ = torch.topk(policy_probs, 2, dim=1)
        policy_probs = top2 / top2.sum(dim=1, keepdim=True)
    M = 0.5 * (policy_probs + value_probs)
    jsd = 0.5 * ((policy_probs * (torch.log(policy_probs+1e-12) - torch.log(M+1e-12))).sum(dim=1)
                 + (value_probs * (torch.log(value_probs+1e-12) - torch.log(M+1e-12))).sum(dim=1))
    epistemic = jsd.cpu().numpy()
    
    # Combined
    combined = alea_w * aleatoric + epis_w * epistemic
    return combined

def generate_uq_csvs(dataset):
    """Generate UQ CSV files for different weight combinations"""
    print(f"\n[PHASE 2] Generating UQ CSV files...")
    
    # Load outputs from Phase 1
    policy_path = f'./test/policy_logits_tta_{dataset}.npy'
    value_path = f'./test/value_preds_{dataset}.csv'
    
    policy_data = np.load(policy_path, allow_pickle=True)
    value_df = pd.read_csv(value_path)
    
    # Convert to tensors
    max_len = max(len(item['agg_scores']) for item in policy_data)
    padded = []
    for item in policy_data:
        scores = item['agg_scores']
        padded_scores = np.pad(scores, (0, max_len-len(scores)), mode='constant')
        padded.append(padded_scores)
    
    policy_tensor = torch.tensor(np.array(padded), dtype=torch.float32)
    value_tensor = torch.tensor(value_df["value"].values, dtype=torch.float32)
    
    print(f"  Policy shape: {policy_tensor.shape}, Value shape: {value_tensor.shape}")
    
    # Generate CSVs for different weights
    os.makedirs('uq_outputs', exist_ok=True)
    weights = [(round(e,1), round(1-e,1)) for e in np.linspace(0.1, 0.9, 9)]
    
    for epis_w, alea_w in weights:
        uq_scores = compute_uq(policy_tensor, value_tensor, alea_w, epis_w)
        filename = f"uq_alea{alea_w:.1f}_epis{epis_w:.1f}.csv"
        filepath = f'./uq_outputs/{filename}'
        pd.DataFrame({"uq_score": uq_scores}).to_csv(filepath, index=False)
        print(f"  ✅ {filename}")
    
    print(f"✅ Generated 9 UQ CSV files")

# ============================================================
# PHASE 3: MCTS WITH UQ
# ============================================================

class MinMaxStats:
    def __init__(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')
    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)
    def normalize(self, value):
        return (value - self.minimum) / (self.maximum - self.minimum) if self.maximum > self.minimum else value

class Node:
    def __init__(self, state, h, prior, cost=0, action_mol=None, fmove=0, reaction=None,
                 template=None, parent=None, cpuct=1.5, uncertainty=0.0):
        self.state, self.h, self.prior, self.cost, self.uncertainty = state, h, prior, cost, uncertainty
        self.action_mol, self.fmove, self.reaction, self.template, self.parent, self.cpuct = action_mol, fmove, reaction, template, parent, cpuct
        self.children, self.visited_time, self.is_expanded, self.child_illegal, self.f_mean_path = [], 0, False, np.array([]), []
        if parent is None:
            self.g, self.depth = 0, 0
        else:
            self.g, self.depth = parent.g + cost, parent.depth + 1
            parent.children.append(self)
        self.f = self.g + self.h
    
    def child_Q(self, stats):
        return np.array([1 - np.mean(stats.normalize(c.f_mean_path)) if c.f_mean_path else 0.0 for c in self.children])
    
    def child_U(self):
        Ns = np.array([c.visited_time for c in self.children]) + 1
        priors = np.array([c.prior for c in self.children])
        return self.cpuct * np.sqrt(self.visited_time) * priors / Ns
    
    def select_child(self, stats, uq_weight=0.5):
        score = self.child_Q(stats) + self.child_U()
        uncertainties = np.array([c.uncertainty for c in self.children])
        if len(uncertainties) > 0 and np.max(uncertainties) > 0:
            score -= uq_weight * (uncertainties / np.max(uncertainties))
        score -= self.child_illegal
        return np.argmax(score)

class MCTS_UQ:
    def __init__(self, target_mol, known_mols, value_model, expand_fn, device,
                 simulations, cpuct, uq_dict, uq_weight):
        self.target_mol, self.known_mols, self.value_model, self.expand_fn = target_mol, known_mols, value_model, expand_fn
        self.device, self.cpuct, self.uq_dict, self.uq_weight = device, cpuct, uq_dict, uq_weight
        self.visited_policy, self.visited_state, self.iterations = {}, [], 0
        self.min_max_stats, self.opening_size = MinMaxStats(), simulations
        
        root_val = value_fn(value_model, [target_mol], device)
        root_uq = uq_dict.get(target_mol, 0.0)
        self.root = Node([target_mol], root_val, 1.0, cpuct=cpuct, uncertainty=root_uq)
        self.min_max_stats.update(self.root.f)
    
    def select_a_leaf(self):
        current = self.root
        while True:
            current.visited_time += 1
            if not current.is_expanded:
                return current
            current = current.children[current.select_child(self.min_max_stats, self.uq_weight)]
    
    def expand(self, node):
        node.is_expanded = True
        mol = node.state[0]
        
        if mol in self.visited_policy:
            policy = self.visited_policy[mol]
        else:
            policy = self.expand_fn.run(mol, topk=50)
            self.iterations += 1
            self.visited_policy[mol] = policy
        
        if not policy or not policy.get('scores'):
            if node.parent:
                node.parent.child_illegal[node.fmove] = 1000
            return False, None
        
        node.child_illegal = np.zeros(len(policy['scores']))
        for i in range(len(policy['scores'])):
            reactants = [r for r in policy['reactants'][i].split('.') if r not in self.known_mols]
            reactants = sorted(set(reactants + node.state[1:]))
            cost = -np.log(np.clip(policy['scores'][i], 1e-3, 1.0))
            
            if not reactants:
                child = Node([], 0, cost=cost, prior=1.0/len(policy['scores']), 
                           action_mol=mol, reaction=policy['reactants'][i]+'>>'+mol,
                           fmove=len(node.children), template=policy['template'][i],
                           parent=node, cpuct=self.cpuct, uncertainty=0.0)
                return True, child
            
            h = value_fn(self.value_model, reactants, self.device)
            uq = max([self.uq_dict.get(r, 0.0) for r in reactants])
            Node(reactants, h, cost=cost, prior=1.0/len(policy['scores']),
                 action_mol=mol, reaction=policy['reactants'][i]+'>>'+mol,
                 fmove=len(node.children), template=policy['template'][i],
                 parent=node, cpuct=self.cpuct, uncertainty=uq)
        
        return False, None
    
    def search(self, max_iter):
        success, node = False, None
        while self.iterations < max_iter and not success:
            if len(self.root.child_illegal) > 0 and np.all(self.root.child_illegal > 0):
                break
            expand_node = [self.select_a_leaf() for _ in range(self.opening_size)][
                np.argmin([n.f for n in [self.select_a_leaf() for _ in range(self.opening_size)]])]
            if '.'.join(expand_node.state) in self.visited_state:
                if expand_node.parent:
                    expand_node.parent.child_illegal[expand_node.fmove] = 1000
                continue
            self.visited_state.append('.'.join(expand_node.state))
            success, node = self.expand(expand_node)
            # Update
            stat = expand_node.f
            self.min_max_stats.update(stat)
            curr = expand_node
            while curr:
                curr.f_mean_path.append(stat)
                curr = curr.parent
        return success, node, self.iterations
    
    def get_path(self, node):
        if not node:
            return [], []
        reactions, templates = [], []
        while node:
            reactions.append(node.reaction)
            templates.append(node.template)
            node = node.parent
        return reactions[::-1], templates[::-1]

def run_mcts_uq(dataset, mols, gpu, device, simulations, cpuct, max_iter, uq_weight):
    """Run MCTS with UQ awareness"""
    print(f"\n[PHASE 3] Running MCTS with UQ...")
    print(f"  Molecules: {len(mols)}, UQ weight: {uq_weight}")
    
    # Load UQ scores
    uq_file = f'./uq_outputs/uq_alea{uq_weight:.1f}_epis{1-uq_weight:.1f}.csv'
    uq_df = pd.read_csv(uq_file)
    uq_dict = {mol: uq for mol, uq in zip(mols, uq_df['uq_score'].values)}
    print(f"  Loaded {len(uq_dict)} UQ scores from {uq_file}")
    
    # Load models
    known_mols = prepare_starting_molecules()
    value_model = prepare_value('./saved_model/value_pc.pt', gpu)
    expand_fn = prepare_expand('./saved_model/policy_model.ckpt', gpu)
    
    # Run MCTS
    results = {'route': [], 'template': [], 'success': [], 'depth': [], 'counts': []}
    for idx, mol in enumerate(mols):
        try:
            with time_limit(600):
                mcts = MCTS_UQ(mol, known_mols, value_model, expand_fn, device,
                             simulations, cpuct, uq_dict, uq_weight)
                success, node, count = mcts.search(max_iter)
                route, template = mcts.get_path(node)
        except:
            success, route, template, node, count = False, [None], [None], None, -1
        
        results['route'].append(route)
        results['template'].append(template)
        results['success'].append(success)
        results['depth'].append(node.depth if success else 32)
        results['counts'].append(count if success else -1)
        
        if (idx+1) % 10 == 0:
            print(f"  {idx+1}/{len(mols)} ({100*(idx+1)/len(mols):.1f}%)")
    
    # Save
    out_file = f'./test/mcts_uq_{dataset}_{simulations}_{cpuct}_{uq_weight}.pkl'
    with open(out_file, 'wb') as f:
        pickle.dump(results, f)
    
    sr = np.mean(results['success'])
    print(f"\n✅ MCTS Results: Success={sr:.3f}, Saved to {out_file}")
    return results

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['1', '2', '3', 'all'], default='all')
    parser.add_argument('--dataset', default='USPTO')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--simulations', type=int, default=100)
    parser.add_argument('--cpuct', type=float, default=4.0)
    parser.add_argument('--max_iterations', type=int, default=500)
    parser.add_argument('--uq_weight', type=float, default=0.5,
                       help='UQ weight (also determines which CSV to load: alea=uq_weight, epis=1-uq_weight)')
    args = parser.parse_args()
    
    os.makedirs('./test', exist_ok=True)
    device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
    
    # Load dataset
    with open(f'./test_dataset/{args.dataset}.pkl', 'rb') as f:
        targets = pickle.load(f)
    print(f"Dataset: {args.dataset} ({len(targets)} molecules)")
    
    # PHASE 1
    if args.phase in ['1', 'all']:
        expand_fn = prepare_expand('./saved_model/policy_model.ckpt', args.gpu)
        value_model = prepare_value('./saved_model/value_pc.pt', args.gpu)
        generate_policy_logits(expand_fn, targets, f'./test/policy_logits_tta_{args.dataset}.npy')
        generate_value_preds(value_model, targets, device, f'./test/value_preds_{args.dataset}.csv')
    
    # PHASE 2
    if args.phase in ['2', 'all']:
        generate_uq_csvs(args.dataset)
    
    # PHASE 3
    if args.phase in ['3', 'all']:
        run_mcts_uq(args.dataset, targets, args.gpu, device, 
                   args.simulations, args.cpuct, args.max_iterations, args.uq_weight)
    
    print(f"\n{'='*60}")
    print("✅ Pipeline Complete!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
