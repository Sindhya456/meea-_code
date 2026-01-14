#!/usr/bin/env python3
"""
Complete MCTS + UQ Pipeline for Lambda Labs
Runs in 3 phases:
1. Generate policy logits & value predictions (TTA)
2. Generate UQ CSV files from the outputs
3. Run MCTS + A* with UQ awareness
"""

import pickle
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import sys
import argparse
import time
import signal
from contextlib import contextmanager
from pathlib import Path

# Import your custom modules
from valueEnsemble import ValueEnsemble
from policyNet import MLPModel
from rdkit import Chem
from rdkit.Chem import AllChem
import heapq

# ============================================================
# UTILITIES
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

class SmilesEnumerator(object):
    def __init__(self, charset='@C)(=cOn1S2/H[N]\\', pad=120,
                 leftpad=True, isomericSmiles=True,
                 enum=True, canonical=False):
        self._charset = None
        self.charset = charset
        self.pad = pad
        self.leftpad = leftpad
        self.isomericSmiles = isomericSmiles
        self.enumerate = enum
        self.canonical = canonical

    @property
    def charset(self):
        return self._charset

    @charset.setter
    def charset(self, charset):
        self._charset = charset
        self._charlen = len(charset)
        self._char_to_int = dict((c, i) for i, c in enumerate(charset))
        self._int_to_char = dict((i, c) for i, c in enumerate(charset))

    def randomize_smiles(self, smiles):
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        return Chem.MolToSmiles(nm, canonical=self.canonical, 
                               isomericSmiles=self.isomericSmiles)

def prepare_starting_molecules(path='./prepare_data/origin_dict.csv'):
    if not os.path.exists(path):
        print(f"[WARN] Starting molecules file not found: {path}")
        return set()
    df = pd.read_csv(path)
    return set(df['mol'].astype(str).tolist()) if 'mol' in df.columns else set()

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
    index = len(fps)
    if index <= 5:
        mask = np.ones(5)
        mask[index:] = 0
        fps_input = np.zeros((5, 2048))
        fps_input[:index] = fps
    else:
        mask = np.ones(index)
        fps_input = fps

    fps = torch.FloatTensor([fps_input]).to(device)
    mask = torch.FloatTensor([mask]).to(device)
    v = model(fps, mask).cpu().data.numpy()
    return v[0][0]

def prepare_value(model_f, gpu):
    device = 'cpu' if gpu == -1 else f'cuda:{gpu}'
    model = ValueEnsemble(2048, 128, 0.1).to(device)
    model.load_state_dict(torch.load(model_f, map_location=device))
    model.eval()
    return model

def prepare_expand(model_path, gpu):
    device = 'cpu' if gpu == -1 else f'cuda:{gpu}'
    return MLPModel(model_path, './saved_model/template_rules.dat', device=device)

# ============================================================
# PHASE 1: GENERATE POLICY LOGITS & VALUE PREDICTIONS
# ============================================================

def _logmeanexp(a, axis=0):
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).mean(axis=axis, keepdims=True))).squeeze(axis)

def save_policy_outputs_tta(expand_fn, mols, device, out_path, 
                            n_aug=20, aggregation="mean", topk=50, save_raw=True):
    """Generate policy logits with TTA"""
    print(f"\n[PHASE 1] Generating policy logits with TTA...")
    print(f"  - Molecules: {len(mols)}")
    print(f"  - Augmentations: {n_aug}")
    print(f"  - Output: {out_path}")
    
    sm_en = SmilesEnumerator(enum=True, canonical=False)
    results = []
    raw_results = []

    with torch.no_grad():
        for idx, s in enumerate(mols):
            aug_candidates = []
            aug_scores = []

            for _ in range(n_aug):
                try:
                    aug_s = sm_en.randomize_smiles(s)
                except Exception:
                    aug_s = s

                out = expand_fn.run(aug_s, topk=topk)
                if out is None:
                    aug_candidates.append([])
                    aug_scores.append(np.array([]))
                    continue

                aug_candidates.append(list(out['reactants']))
                aug_scores.append(np.array(out['scores'], dtype=np.float32))

            union = []
            for cands in aug_candidates:
                for c in cands:
                    if c not in union:
                        union.append(c)

            if len(union) == 0:
                continue

            aligned = np.zeros((n_aug, len(union)), dtype=np.float32)
            for i, (cands, scores) in enumerate(zip(aug_candidates, aug_scores)):
                for j, c in enumerate(cands):
                    aligned[i, union.index(c)] = scores[j]

            if aggregation == "mean":
                agg_scores = aligned.mean(axis=0)
            else:
                log_aligned = np.log(np.clip(aligned, 1e-12, 1.0))
                agg_scores = np.exp(_logmeanexp(log_aligned, axis=0))

            results.append({
                "mol": s,
                "candidates": union,
                "agg_scores": agg_scores
            })

            if save_raw:
                raw_results.append({
                    "mol": s,
                    "union_candidates": union,
                    "aligned_scores": aligned
                })

            if (idx + 1) % 10 == 0:
                print(f"  Progress: {idx+1}/{len(mols)} ({100*(idx+1)/len(mols):.1f}%)")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, np.array(results, dtype=object))
    if save_raw:
        np.save(out_path.replace(".npy", "_raw.npy"), np.array(raw_results, dtype=object))
    
    print(f"✅ Saved policy logits to {out_path}")

def save_value_outputs(value_model, mols, device, out_path):
    """Generate value predictions"""
    print(f"\n[PHASE 1] Generating value predictions...")
    print(f"  - Molecules: {len(mols)}")
    print(f"  - Output: {out_path}")
    
    preds = []
    for idx, s in enumerate(mols):
        pred = value_fn(value_model, [s], device)
        preds.append(pred)
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx+1}/{len(mols)} ({100*(idx+1)/len(mols):.1f}%)")
    
    df = pd.DataFrame({"mol": mols, "value": preds})
    df.to_csv(out_path, index=False)
    print(f"✅ Saved value predictions to {out_path}")

# ============================================================
# PHASE 2: GENERATE UQ CSV FILES
# ============================================================

def compute_aleatoric_uncertainty(policy_logits_tta):
    """Compute aleatoric uncertainty (predictive entropy)"""
    probs = F.softmax(policy_logits_tta, dim=1)
    mean_probs = probs.mean(dim=0)
    entropy = -(mean_probs * torch.log(mean_probs + 1e-12)).sum().item()
    return entropy, mean_probs

def compute_epistemic_uncertainty_jsd(policy_logits, value_preds):
    """Compute epistemic uncertainty via JSD"""
    policy_probs = F.softmax(policy_logits, dim=1)
    value_probs = torch.sigmoid(value_preds).squeeze(-1)
    value_probs = torch.stack([value_probs, 1 - value_probs], dim=1)

    if policy_probs.shape[1] != 2:
        top2_probs, _ = torch.topk(policy_probs, 2, dim=1)
        policy_probs = top2_probs / top2_probs.sum(dim=1, keepdim=True)

    M = 0.5 * (policy_probs + value_probs)
    jsd = 0.5 * (
        (policy_probs * (torch.log(policy_probs + 1e-12) - torch.log(M + 1e-12))).sum(dim=1)
        + (value_probs * (torch.log(value_probs + 1e-12) - torch.log(M + 1e-12))).sum(dim=1)
    )
    return jsd.cpu().numpy()

def compute_combined_uq(aleatoric, epistemic, method="weighted_sum",
                       alea_weight=0.5, epis_weight=0.5):
    """Combine aleatoric + epistemic"""
    aleatoric = np.asarray(aleatoric).ravel()
    epistemic = np.asarray(epistemic).ravel()
    if method == "weighted_sum":
        combined = alea_weight * aleatoric + epis_weight * epistemic
    elif method == "geometric_mean":
        total = alea_weight + epis_weight
        w_alea = alea_weight / total
        w_epis = epis_weight / total
        combined = np.power(aleatoric, w_alea) * np.power(epistemic, w_epis)
    else:
        raise ValueError(f"Unsupported combination method: {method}")
    return combined.ravel()

def uq_analysis(policy_logits, value_preds, alea_weight=0.5, epis_weight=0.5, 
               method="weighted_sum"):
    """Compute all UQ metrics"""
    B, C = policy_logits.shape

    aleatoric = []
    for i in range(B):
        ent, _ = compute_aleatoric_uncertainty(policy_logits[i].unsqueeze(0))
        aleatoric.append(ent)
    aleatoric = np.array(aleatoric)

    epistemic = compute_epistemic_uncertainty_jsd(policy_logits, value_preds)

    uq_scores = compute_combined_uq(aleatoric, epistemic, method=method,
                                   alea_weight=alea_weight, epis_weight=epis_weight)

    return {
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "combined": uq_scores
    }

def generate_uq_files(dataset, policy_logits_path, value_preds_path, save_dir="uq_outputs"):
    """Generate UQ CSV files for different weight combinations"""
    print(f"\n[PHASE 2] Generating UQ files...")
    print(f"  - Policy logits: {policy_logits_path}")
    print(f"  - Value preds: {value_preds_path}")
    
    # Load data
    policy_logits = np.load(policy_logits_path, allow_pickle=True)
    value_df = pd.read_csv(value_preds_path)
    
    # Convert to tensors
    max_len = max(len(item['agg_scores']) for item in policy_logits)
    padded = []
    for item in policy_logits:
        scores = item['agg_scores']
        pad_len = max_len - len(scores)
        padded_scores = np.pad(scores, (0, pad_len), mode='constant')
        padded.append(padded_scores)
    
    policy_logits_tensor = torch.tensor(np.array(padded), dtype=torch.float32)
    value_preds_tensor = torch.tensor(value_df["value"].values, dtype=torch.float32)
    
    print(f"  - Policy logits shape: {policy_logits_tensor.shape}")
    print(f"  - Value preds shape: {value_preds_tensor.shape}")
    
    os.makedirs(save_dir, exist_ok=True)
    weights = [(round(e, 1), round(1 - e, 1)) for e in np.linspace(0.1, 0.9, 9)]
    
    for epis_weight, alea_weight in weights:
        result = uq_analysis(policy_logits_tensor, value_preds_tensor,
                           alea_weight=alea_weight, epis_weight=epis_weight,
                           method="weighted_sum")
        
        uq_scores = np.array(result["combined"]).ravel()
        filename = f"uq_alea{alea_weight:.1f}_epis{epis_weight:.1f}.csv"
        filepath = os.path.join(save_dir, filename)
        pd.DataFrame({"uq_score": uq_scores}).to_csv(filepath, index=False)
        print(f"  ✅ Saved: {filename}")
    
    print(f"✅ All UQ files generated in {save_dir}/")

# ============================================================
# PHASE 3: MCTS + A* WITH UQ
# ============================================================

def load_uncertainty_data(dataset, save_dir="uq_outputs", alea_weight=0.5, epis_weight=0.5):
    """Load UQ scores from CSV"""
    filename = f"uq_alea{alea_weight:.1f}_epis{epis_weight:.1f}.csv"
    filepath = os.path.join(save_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"[WARN] UQ file not found: {filepath}, using zero uncertainty")
        return {}
    
    df = pd.read_csv(filepath)
    
    # Load original dataset to get SMILES
    dataset_path = f'./test_dataset/{dataset}.pkl'
    with open(dataset_path, 'rb') as f:
        targets = pickle.load(f)
    
    # Create dictionary mapping SMILES to UQ scores
    uncertainty_dict = {}
    for smiles, uq_score in zip(targets, df['uq_score'].values):
        uncertainty_dict[smiles] = float(uq_score)
    
    print(f"[INFO] Loaded {len(uncertainty_dict)} uncertainty values from {filepath}")
    return uncertainty_dict

class MinMaxStats:
    def __init__(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

class Node:
    def __init__(self, state, h, prior, cost=0, action_mol=None, fmove=0, 
                 reaction=None, template=None, parent=None, cpuct=1.5, uncertainty=0.0):
        self.state = state
        self.h = h
        self.prior = prior
        self.cost = cost
        self.uncertainty = uncertainty
        self.action_mol = action_mol
        self.fmove = fmove
        self.reaction = reaction
        self.template = template
        self.parent = parent
        self.cpuct = cpuct
        self.children = []
        self.visited_time = 0
        self.is_expanded = False
        self.child_illegal = np.array([])
        self.f_mean_path = []

        if parent is None:
            self.g = 0
            self.depth = 0
        else:
            self.g = parent.g + cost
            self.depth = parent.depth + 1
            parent.children.append(self)

        self.f = self.g + self.h

    def child_N(self):
        return np.array([child.visited_time for child in self.children])

    def child_p(self):
        return np.array([child.prior for child in self.children])

    def child_U(self):
        child_Ns = self.child_N() + 1
        prior = self.child_p()
        return self.cpuct * np.sqrt(self.visited_time) * prior / child_Ns

    def child_uncertainty(self):
        return np.array([child.uncertainty for child in self.children])

    def child_Q(self, min_max_stats):
        child_Qs = []
        for child in self.children:
            if len(child.f_mean_path) == 0:
                child_Qs.append(0.0)
            else:
                child_Qs.append(1 - np.mean(min_max_stats.normalize(child.f_mean_path)))
        return np.array(child_Qs)

    def select_child(self, min_max_stats, uncertainty_weight=0.5):
        action_score = self.child_Q(min_max_stats) + self.child_U()
        
        child_uncertainties = self.child_uncertainty()
        if len(child_uncertainties) > 0 and np.max(child_uncertainties) > 0:
            normalized_uncertainty = child_uncertainties / np.max(child_uncertainties)
            action_score -= uncertainty_weight * normalized_uncertainty
        
        action_score -= self.child_illegal
        return np.argmax(action_score)

class MCTS_A:
    def __init__(self, target_mol, known_mols, value_model, expand_fn, device,
                 simulations, cpuct, uncertainty_dict=None, uncertainty_weight=0.5):
        self.target_mol = target_mol
        self.known_mols = known_mols
        self.expand_fn = expand_fn
        self.value_model = value_model
        self.device = device
        self.cpuct = cpuct
        self.uncertainty_dict = uncertainty_dict if uncertainty_dict else {}
        self.uncertainty_weight = uncertainty_weight

        root_value = value_fn(value_model, [target_mol], device)
        root_uncertainty = self.uncertainty_dict.get(target_mol, 0.0)

        self.root = Node([target_mol], root_value, prior=1.0, cpuct=cpuct,
                        uncertainty=root_uncertainty)
        self.visited_policy = {}
        self.visited_state = []
        self.min_max_stats = MinMaxStats()
        self.min_max_stats.update(self.root.f)
        self.opening_size = simulations
        self.iterations = 0

    def select_a_leaf(self):
        current = self.root
        while True:
            current.visited_time += 1
            if not current.is_expanded:
                return current
            best_move = current.select_child(self.min_max_stats, self.uncertainty_weight)
            current = current.children[best_move]

    def select(self):
        openings = [self.select_a_leaf() for _ in range(self.opening_size)]
        stats = [opening.f for opening in openings]
        return openings[np.argmin(stats)]

    def get_state_uncertainty(self, reactant_list):
        if len(reactant_list) == 0:
            return 0.0
        uncertainties = [self.uncertainty_dict.get(mol, 0.0) for mol in reactant_list]
        return np.max(uncertainties)

    def expand(self, node):
        node.is_expanded = True
        expanded_mol = node.state[0]

        if expanded_mol in self.visited_policy:
            expanded_policy = self.visited_policy[expanded_mol]
        else:
            expanded_policy = self.expand_fn.run(expanded_mol, topk=50)
            self.iterations += 1
            self.visited_policy[expanded_mol] = expanded_policy.copy() if expanded_policy else None

        if expanded_policy is not None and len(expanded_policy['scores']) > 0:
            node.child_illegal = np.array([0] * len(expanded_policy['scores']))

            for i in range(len(expanded_policy['scores'])):
                reactant = [r for r in expanded_policy['reactants'][i].split('.')
                           if r not in self.known_mols]
                reactant = reactant + node.state[1:]
                reactant = sorted(list(set(reactant)))
                cost = -np.log(np.clip(expanded_policy['scores'][i], 1e-3, 1.0))
                template = expanded_policy['template'][i]
                reaction = expanded_policy['reactants'][i] + '>>' + expanded_mol
                priors = np.array([1.0 / len(expanded_policy['scores'])] * len(expanded_policy['scores']))

                if len(reactant) == 0:
                    child = Node([], 0, cost=cost, prior=priors[i], action_mol=expanded_mol,
                               reaction=reaction, fmove=len(node.children), template=template,
                               parent=node, cpuct=self.cpuct, uncertainty=0.0)
                    return True, child
                else:
                    h = value_fn(self.value_model, reactant, self.device)
                    state_uncertainty = self.get_state_uncertainty(reactant)
                    child = Node(reactant, h, cost=cost, prior=priors[i],
                               action_mol=expanded_mol, reaction=reaction,
                               fmove=len(node.children), template=template,
                               parent=node, cpuct=self.cpuct, uncertainty=state_uncertainty)

                    if '.'.join(reactant) in self.visited_state:
                        node.child_illegal[child.fmove] = 1000
        else:
            if node.parent is not None:
                node.parent.child_illegal[node.fmove] = 1000
        
        return False, None

    def update(self, node):
        stat = node.f
        self.min_max_stats.update(stat)
        current = node
        while current is not None:
            current.f_mean_path.append(stat)
            current = current.parent

    def search(self, times):
        success, node = False, None
        while self.iterations < times and not success:
            if len(self.root.child_illegal) > 0 and np.all(self.root.child_illegal > 0):
                break
            
            expand_node = self.select()
            if '.'.join(expand_node.state) in self.visited_state:
                if expand_node.parent:
                    expand_node.parent.child_illegal[expand_node.fmove] = 1000
                continue
            
            self.visited_state.append('.'.join(expand_node.state))
            success, node = self.expand(expand_node)
            self.update(expand_node)
            
            if self.visited_policy.get(self.target_mol) is None:
                return False, None, times
        
        return success, node, self.iterations

    def vis_synthetic_path(self, node):
        if node is None:
            return [], []
        reaction_path, template_path = [], []
        current = node
        while current is not None:
            reaction_path.append(current.reaction)
            template_path.append(current.template)
            current = current.parent
        return reaction_path[::-1], template_path[::-1]

def play_mcts(dataset, mols, known_mols, value_model, expand_fn, device,
              simulations, cpuct, times, uncertainty_dict, uncertainty_weight=0.5):
    """Run MCTS with UQ"""
    print(f"\n[PHASE 3] Running MCTS with UQ...")
    print(f"  - Molecules: {len(mols)}")
    print(f"  - Simulations: {simulations}")
    print(f"  - Uncertainty weight: {uncertainty_weight}")
    
    routes, templates, successes, depths, counts = [], [], [], [], []

    for idx, mol in enumerate(mols):
        try:
            with time_limit(600):
                player = MCTS_A(mol, known_mols, value_model, expand_fn, device,
                               simulations, cpuct, uncertainty_dict, uncertainty_weight)
                success, node, count = player.search(times)
                route, template = player.vis_synthetic_path(node)
        except:
            success = False
            route, template = [None], [None]
        
        routes.append(route)
        templates.append(template)
        successes.append(success)
        depths.append(node.depth if success else 32)
        counts.append(count if success else -1)
        
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx+1}/{len(mols)} ({100*(idx+1)/len(mols):.1f}%)")

    result = {
        'route': routes,
        'template': templates,
        'success': successes,
        'depth': depths,
        'counts': counts
    }

    out_file = f'./test/stat_mcts_uq_{dataset}_{simulations}_{cpuct}_{uncertainty_weight}.pkl'
    with open(out_file, 'wb') as f:
        pickle.dump(result, f, protocol=4)
    
    success_rate = np.mean(successes)
    avg_depth = np.mean(depths)
    print(f"\n✅ MCTS Results:")
    print(f"  - Success rate: {success_rate:.3f}")
    print(f"  - Avg depth: {avg_depth:.2f}")
    print(f"  - Saved to: {out_file}")
    
    return result

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Complete MCTS + UQ Pipeline for Lambda Labs')
    parser.add_argument('--phase', type=str, choices=['1', '2', '3', 'all'], default='all',
                       help='Which phase to run (1=generate, 2=uq, 3=mcts, all=everything)')
    parser.add_argument('--dataset', type=str, default='USPTO',
                       help='Dataset name')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU id')
    parser.add_argument('--simulations', type=int, default=100,
                       help='MCTS simulations')
    parser.add_argument('--cpuct', type=float, default=4.0,
                       help='MCTS exploration constant')
    parser.add_argument('--max_iterations', type=int, default=500,
                       help='Max iterations')
    parser.add_argument('--uncertainty_weight', type=float, default=0.5,
                       help='Uncertainty weight (0-1)')
    parser.add_argument('--alea_weight', type=float, default=0.5,
                       help='Aleatoric weight for UQ combination')
    parser.add_argument('--epis_weight', type=float, default=0.5,
                       help='Epistemic weight for UQ combination')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs('./test', exist_ok=True)
    os.makedirs('./uq_outputs', exist_ok=True)
    
    device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
    print(f"\n{'='*60}")
    print(f"MCTS + UQ Pipeline")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    # Load dataset
    dataset_path = f'./test_dataset/{args.dataset}.pkl'
    with open(dataset_path, 'rb') as f:
        targets = pickle.load(f)
    print(f"Loaded {len(targets)} target molecules")
    
    # PHASE 1: Generate policy logits & value preds
    if args.phase in ['1', 'all']:
        print(f"\n{'='*60}")
        print("PHASE 1: Generate Policy Logits & Value Predictions")
        print(f"{'='*60}")
        
        expand_fn = prepare_expand('./saved_model/policy_model.ckpt', args.gpu)
        value_model = prepare_value('./saved_model/value_pc.pt', args.gpu)
        
        save_policy_outputs_tta(
            expand_fn, targets, device,
            out_path=f'./test/policy_logits_tta_{args.dataset}.npy',
            n_aug=20, aggregation="mean", topk=50, save_raw=True
        )
        
        save_value_outputs(
            value_model, targets, device,
            out_path=f'./
