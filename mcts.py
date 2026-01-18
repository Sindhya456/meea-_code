#!/usr/bin/env python3
# full_pipeline_meea.py
# End-to-end 3-phase MEEA pipeline: Phase1 (TTA), Phase2 (UQ/weighting), Phase3 (weighted MCTS+A*)

import os, sys, pickle, time, heapq
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from rdkit import Chem
from rdkit.Chem import AllChem
from glob import glob
import multiprocessing

from valueEnsemble import ValueEnsemble
from policyNet import MLPModel

# ---------------------------
# GPU & device management
# ---------------------------
device_list = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
print(f"[INFO] Using {len(device_list)} GPU(s): {device_list}")

# ---------------------------
# Smiles Enumerator for TTA
# ---------------------------
class SmilesEnumerator:
    def __init__(self, charset='@C)(=cOn1S2/H[N]\\', pad=120, isomericSmiles=True, enum=True, canonical=False):
        self._charset = charset
        self.pad = pad
        self.isomericSmiles = isomericSmiles
        self.enumerate = enum
        self.canonical = canonical

    def randomize_smiles(self, smiles):
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        idx = list(range(m.GetNumAtoms()))
        np.random.shuffle(idx)
        nm = Chem.RenumberAtoms(m, idx)
        return Chem.MolToSmiles(nm, canonical=self.canonical, isomericSmiles=self.isomericSmiles)

# ---------------------------
# Fingerprint helpers
# ---------------------------
def smiles_to_fp(s, fp_dim=2048):
    mol = Chem.MolFromSmiles(str(s))
    if mol is None:
        return np.zeros(fp_dim, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    arr = np.zeros(fp.GetNumBits(), dtype=np.float32)
    onbits = list(fp.GetOnBits())
    arr[onbits] = 1
    return arr

def batch_smiles_to_fp(s_list, fp_dim=2048):
    return np.array([smiles_to_fp(s, fp_dim) for s in s_list], dtype=np.float32)


# -------------------------
# Model loaders
# -------------------------
import os

def prepare_expand(model_path, gpu=None):
    device = 'cpu' if gpu is None else gpu
    template_path = os.path.expanduser('~/content/template_rules.dat')
    one_step = MLPModel(
        model_path,
        template_path,
        device=device
    )
    return one_step



def prepare_value(model_f, gpu=None):
    device = 'cpu' if gpu is None else gpu
    model = ValueEnsemble(2048, 128, 0.1).to(device)
    ckpt = torch.load(model_f, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    return model

# ---------------------------
# Phase 1: TTA Policy + Value Predictions
# ---------------------------
def save_policy_outputs_tta(expand_fn, mols, device="cuda:0", out_path="policy_logits_tta.npy",
                            n_aug=20, aggregation="mean", topk=50):
    sm_en = SmilesEnumerator(enum=True, canonical=False)
    results = []

    with torch.no_grad():
        for idx, mol in enumerate(mols):
            aug_scores_list = []
            aug_cands_list = []
            for _ in range(n_aug):
                try:
                    s = sm_en.randomize_smiles(mol)
                except:
                    s = mol
                out = expand_fn.run(s, topk=topk)
                if out is None:
                    aug_scores_list.append(np.array([]))
                    aug_cands_list.append([])
                    continue
                aug_scores_list.append(np.array(out['scores'], dtype=np.float32))
                aug_cands_list.append(out['reactants'])

            # union of candidates
            union_set = []
            for cands in aug_cands_list:
                for c in cands:
                    if c not in union_set:
                        union_set.append(c)
            if len(union_set) == 0:
                continue

            aligned = np.zeros((len(aug_scores_list), len(union_set)), dtype=np.float32)
            for i, (cands, scores) in enumerate(zip(aug_cands_list, aug_scores_list)):
                for j, c in enumerate(cands):
                    idx_j = union_set.index(c)
                    aligned[i, idx_j] = scores[j]

            # aggregate
            if aggregation=="mean":
                agg_scores = aligned.mean(axis=0)
            elif aggregation=="logmeanexp":
                m = np.max(aligned, axis=0, keepdims=True)
                agg_scores = (m + np.log(np.exp(aligned - m).mean(axis=0, keepdims=True))).squeeze()
            else:
                raise ValueError("Unknown aggregation")
            results.append({'mol': mol, 'candidates': union_set, 'agg_scores': agg_scores})

            if (idx + 1) % 100 == 0:
                print(f"[INFO] Phase1 processed {idx+1}/{len(mols)} molecules")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, np.array(results, dtype=object))
    print(f"[INFO] Phase1 Policy logits saved: {out_path}")
    return results

def save_value_outputs(value_model, mols, device="cuda:0", out_path="value_preds.csv"):
    preds = []
    with torch.no_grad():
        for idx, mol in enumerate(mols):
            fp = torch.from_numpy(batch_smiles_to_fp([mol])).unsqueeze(0).to(device)
            mask = torch.ones((1,1), device=device)
            v = value_model(fp, mask).cpu().item()
            preds.append(v)
            if (idx + 1) % 100 == 0:
                print(f"[INFO] Phase1 Value processed {idx+1}/{len(mols)} molecules")
    df = pd.DataFrame({'mol': mols, 'value': preds})
    df.to_csv(out_path, index=False)
    print(f"[INFO] Phase1 Value predictions saved: {out_path}")
    return preds

# ---------------------------
# Phase 2: Epistemic UQ weighting (JS divergence)
# ---------------------------
def js_divergence(p, q):
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div(torch.tensor(m).log(), torch.tensor(p), reduction='sum') +
                  F.kl_div(torch.tensor(m).log(), torch.tensor(q), reduction='sum')).item()

def compute_weights(policy_logits):
    weights = []
    for mol_obj in policy_logits:
        aug_scores = mol_obj.get('agg_scores')
        if aug_scores is None:
            weights.append(1.0)
            continue
        p = torch.tensor(aug_scores)
        p = p / p.sum()
        m = torch.mean(p)
        weight = 1.0 + float(torch.mean(torch.abs(p - m)))
        weights.append(weight)
    return weights

# ---------------------------
# Phase 3: Weighted MCTS + A* search
# ---------------------------
class AStarNode:
    def __init__(self, state, g, h, action=None, parent=None):
        self.state = state
        self.g = g
        self.h = h
        self.f = g + h
        self.action = action
        self.parent = parent
    def __lt__(self, other):
        return self.f < other.f

def meea_star(target_mol, known_mols, value_model, expand_fn, weight=1.0, device="cuda:0", max_steps=500):
    start = time.time()
    root_h = 1.0
    root = AStarNode([target_mol], g=0.0, h=root_h)
    open_list = []
    heapq.heappush(open_list, root)
    visited = set()
    expansions = 0
    while open_list and expansions < max_steps:
        node = heapq.heappop(open_list)
        expansions += 1
        if all(m in known_mols for m in node.state):
            return True, node, expansions, time.time()-start
        mol = node.state[0]
        out = expand_fn.run(mol, topk=50)
        if out is None:
            continue
        scores = out['scores']
        reactants_list = out['reactants']
        templates = out.get('template', [None]*len(scores))
        for i in range(len(scores)):
            reactants = [r for r in reactants_list[i].split('.') if r not in known_mols and r != '']
            new_state = sorted(list(set(reactants + node.state[1:])))
            key = '.'.join(new_state)
            if key in visited:
                continue
            visited.add(key)
            cost = -np.log(max(scores[i], 1e-6)) * (2.0 - weight)
            h = 0.0
            child = AStarNode(new_state, g=node.g+cost, h=h, parent=node, action=(templates[i], reactants_list[i]))
            heapq.heappush(open_list, child)
    return False, None, expansions, time.time()-start

def play_meea(dataset, mols, known_mols, value_model, expand_fn, weights=None, device="cuda:0"):
    results = []
    for idx, mol in enumerate(mols):
        w = weights[idx] if weights else 1.0
        success, node, expansions, elapsed = meea_star(mol, known_mols, value_model, expand_fn, weight=w, device=device)
        depth = node.g if success else 32
        results.append({'success': success, 'depth': depth, 'expansions': expansions, 'time': elapsed})
        sys.stdout.write(f"\rDataset {dataset}: {idx+1}/{len(mols)} molecules done")
        sys.stdout.flush()
    print()
    success_rate = np.mean([r['success'] for r in results])
    avg_depth = np.mean([r['depth'] for r in results if r['success']]) if any(r['success'] for r in results) else 0
    avg_exp = np.mean([r['expansions'] for r in results])
    avg_time = np.mean([r['time'] for r in results])
    return success_rate, avg_depth, avg_exp, avg_time, results

# ---------------------------
# Main pipeline
# ---------------------------
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    test_folder = './test_dataset/'
    out_folder = './test/'
    os.makedirs(out_folder, exist_ok=True)

    known_file = './origin_dict.csv'
    if os.path.isfile(known_file):
        known_mols = set(pd.read_csv(known_file)['mol'].astype(str))
    else:
        known_mols = set()

    policy_model_path = '/home/ubuntu/meea-_code/saved_model/policy_model.ckpt'

    value_model_path = '/home/ubuntu/meea-_code/saved_model/value_pc.pt'
    expand_fn = prepare_expand(policy_model_path, gpu=device_list[0])
    value_model = prepare_value(value_model_path, gpu=device_list[0])

    pkl_files = sorted(glob(os.path.join(test_folder, '*.pkl')))
    for pkl_path in pkl_files:
        dataset_name = os.path.basename(pkl_path).replace('.pkl','')
        print(f"\n===== Processing dataset {dataset_name} =====")

        with open(pkl_path,'rb') as f:
            raw_obj = pickle.load(f)

        if isinstance(raw_obj, list):
            targets = [str(x) for x in raw_obj]
        elif isinstance(raw_obj, pd.DataFrame):
            if 'reaction' in raw_obj.columns:
                targets = raw_obj['reaction'].astype(str).tolist()
            else:
                targets = [str(x) for x in raw_obj.iloc[:,0]]
        else:
            targets = list(raw_obj.values())

        # Phase 1
        policy_logits = save_policy_outputs_tta(expand_fn, targets, device=device_list[0],
                                                out_path=os.path.join(out_folder,f'policy_logits_tta_{dataset_name}.npy'))
        value_preds = save_value_outputs(value_model, targets, device=device_list[0],
                                        out_path=os.path.join(out_folder,f'value_preds_{dataset_name}.csv'))

        # Phase 2
        weights = compute_weights(policy_logits)
        with open(os.path.join(out_folder,f'weights_{dataset_name}.pkl'),'wb') as f:
            pickle.dump(weights, f)
        print(f"[INFO] Phase2 weights saved: weights_{dataset_name}.pkl")

        # Phase 3
        success_rate, avg_depth, avg_exp, avg_time, results = play_meea(dataset_name, targets,
                                                                        known_mols, value_model, expand_fn,
                                                                        weights=weights, device=device_list[0])

        out = {'success_rate': success_rate,
               'avg_depth': avg_depth,
               'avg_expansions': avg_exp,
               'avg_time': avg_time,
               'details': results}
        with open(os.path.join(out_folder,f'stat_meea_{dataset_name}.pkl'),'wb') as f:
            pickle.dump(out,f)

        print(f"\n📊 Dataset {dataset_name} completed")
        print(f"  Success rate: {success_rate:.3f}")
        print(f"  Avg depth: {avg_depth:.2f}")
        print(f"  Avg expansions: {avg_exp:.2f}")
        print(f"  Avg time: {avg_time:.2f}s")
        print("="*50)

    print("\n🎯 All datasets processed successfully.")
