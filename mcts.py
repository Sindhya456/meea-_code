#!/usr/bin/env python3
# =============================================
# Full MEEA Pipeline: Phase1 → Phase2 → Phase3
# GPU-Aware | UQ + Weighted Dataset + Stats
# =============================================
import os
import sys
import time
import heapq
import pickle
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem

# -------------------------
# GPU device
# -------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")

# -------------------------
# -------------------------
# Phase 1: Baseline MCTS + A* → Policy logits & Value predictions
# -------------------------
class SmilesEnumerator:
    def __init__(self, charset='@C)(=cOn1S2/H[N]\\', pad=120, leftpad=True, isomericSmiles=True, enum=True, canonical=False):
        self.charset = charset
        self.pad = pad
        self.leftpad = leftpad
        self.isomericSmiles = isomericSmiles
        self.enumerate = enum
        self.canonical = canonical

    def randomize_smiles(self, smiles):
        m = Chem.MolFromSmiles(smiles)
        if m is None: return smiles
        idxs = list(range(m.GetNumAtoms()))
        np.random.shuffle(idxs)
        nm = Chem.RenumberAtoms(m, idxs)
        return Chem.MolToSmiles(nm, canonical=self.canonical, isomericSmiles=self.isomericSmiles)

# -------------------------
# Fingerprint helpers
# -------------------------
def smiles_to_fp(s, fp_dim=2048):
    if isinstance(s, dict):
        s = s.get("reaction", "")
    s = str(s or "").strip()
    mol = Chem.MolFromSmiles(s)
    if mol is None: return np.zeros(fp_dim, dtype=np.bool_)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool_)
    arr[list(fp.GetOnBits())] = 1
    return arr

def batch_smiles_to_fp(s_list, fp_dim=2048):
    return np.array([smiles_to_fp(s, fp_dim) for s in s_list], dtype=np.float32)

# -------------------------
# Value function
# -------------------------
def value_fn(model, mols, device=device):
    fps = batch_smiles_to_fp(mols)
    mask = np.ones(len(fps), dtype=np.float32)
    fps_tensor = torch.from_numpy(fps).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        v = model(fps_tensor, mask_tensor).cpu().numpy()
    return float(v.flatten()[0])

# -------------------------
# Phase 2: Uncertainty (Aleatoric + Epistemic) & Weighted Dataset
# -------------------------
def compute_aleatoric(policy_logits_tta):
    probs = F.softmax(policy_logits_tta, dim=1)
    mean_probs = probs.mean(dim=0)
    entropy = -(mean_probs * torch.log(mean_probs + 1e-12)).sum().item()
    return entropy

def compute_epistemic_jsd(policy_logits, value_preds):
    policy_probs = F.softmax(policy_logits, dim=1)
    value_probs = torch.sigmoid(value_preds).squeeze(-1)
    value_probs = torch.stack([value_probs, 1 - value_probs], dim=1)
    if policy_probs.shape[1] != 2:
        top2, _ = torch.topk(policy_probs, 2, dim=1)
        policy_probs = top2 / top2.sum(dim=1, keepdim=True)
    M = 0.5 * (policy_probs + value_probs)
    jsd = 0.5 * ((policy_probs * (torch.log(policy_probs + 1e-12) - torch.log(M + 1e-12))).sum(dim=1) +
                 (value_probs * (torch.log(value_probs + 1e-12) - torch.log(M + 1e-12))).sum(dim=1))
    return jsd.cpu().numpy()

def compute_combined_uq(alea, epis, alea_w=0.5, epis_w=0.5):
    return (alea_w * np.array(alea) + epis_w * np.array(epis)).ravel()

def generate_uq_files(policy_logits, value_preds, save_dir='uq_outputs', required_len=None):
    os.makedirs(save_dir, exist_ok=True)
    weights = [(round(e, 1), round(1-e, 1)) for e in np.linspace(0.1, 0.9, 9)]
    B = policy_logits.shape[0]
    if required_len and B != required_len:
        pad_len = required_len - B
        if pad_len > 0:
            policy_logits = torch.cat([policy_logits, torch.zeros(pad_len, policy_logits.shape[1], device=device)], dim=0)
            value_preds = torch.cat([value_preds, torch.zeros(pad_len, 1, device=device)], dim=0)
    for alea_w, epis_w in weights:
        alea = [compute_aleatoric(policy_logits[i].unsqueeze(0)) for i in range(required_len)]
        epis = compute_epistemic_jsd(policy_logits, value_preds)
        combined = compute_combined_uq(alea, epis, alea_w, epis_w)
        filename = f"uq_alea{alea_w:.1f}_epis{epis_w:.1f}.csv"
        pd.DataFrame({"uq_score": combined}).to_csv(os.path.join(save_dir, filename), index=False)
        print(f"[INFO] Saved {filename}")

def reweight_dataset(df, uq_csv, temp=0.4):
    uq = pd.read_csv(uq_csv)['uq_score'].to_numpy().ravel()
    weights = np.exp(-uq / temp)
    weights /= weights.sum()
    df['weight'] = weights.astype(np.float32)
    return df

# -------------------------
# Phase 3: MCTS + A* (UQ aware)
# -------------------------
class AStarNode:
    def __init__(self, state, g, h, action=None, parent=None):
        self.state = state
        self.g, self.h = g, h
        self.f = g + h
        self.action, self.parent = action, parent
    def __lt__(self, other): return self.f < other.f

def meea_star(target_mol, known_mols, value_model, expand_fn, device, max_steps=500):
    weight = float(target_mol.get("weight", 1.0)) if isinstance(target_mol, dict) else 1.0
    target_smiles = target_mol.get("reaction", "") if isinstance(target_mol, dict) else target_mol
    start_time = time.time()
    root_h = value_fn(value_model, [target_smiles], device)
    root = AStarNode([target_smiles], g=0.0, h=root_h)
    open_list, visited, expansions = [], set(), 0
    heapq.heappush(open_list, root)
    while open_list and expansions < max_steps:
        node = heapq.heappop(open_list)
        expansions += 1
        if all(m in known_mols for m in node.state):
            return True, node, expansions, time.time() - start_time
        expanded_mol = node.state[0]
        expanded_policy = expand_fn.run(expanded_mol, topk=50)
        if not expanded_policy: continue
        for i, r in enumerate(expanded_policy['reactants']):
            reactants = [x for x in r.split('.') if x not in known_mols and x != '']
            new_state = sorted(list(set(reactants + node.state[1:])))
            key = '.'.join(new_state)
            if key in visited: continue
            visited.add(key)
            score_val = float(np.clip(float(expanded_policy['scores'][i]), 1e-6, 1.0))
            cost = -np.log(score_val) * (2.0 - weight)
            h = value_fn(value_model, new_state, device) if new_state else 0.0
            child = AStarNode(new_state, g=node.g + cost, h=h, parent=node,
                              action=(expanded_policy['template'][i] if i < len(expanded_policy['template']) else None, r))
            heapq.heappush(open_list, child)
    return False, None, expansions, time.time() - start_time

def play_meea(dataset, mols, known_mols, value_model, expand_fn, device):
    results = []
    for i, mol in enumerate(mols):
        success, node, expansions, elapsed = meea_star(mol, known_mols, value_model, expand_fn, device)
        depth = node.g if success else 32
        results.append({"success": success, "depth": depth, "expansions": expansions, "time": elapsed})
        sys.stdout.write(f"\r[{dataset}] {i+1}/{len(mols)} molecules done")
        sys.stdout.flush()
    print()
    success_rate = np.mean([r['success'] for r in results])
    avg_depth = np.mean([r['depth'] for r in results if r['success']]) if results else float('nan')
    avg_exp = np.mean([r['expansions'] for r in results]) if results else 0.0
    avg_time = np.mean([r['time'] for r in results]) if results else 0.0
    return success_rate, avg_depth, avg_exp, avg_time, results

# -------------------------
# -------------------------
# Main pipeline
# -------------------------
def main():
    # --- Load USPTO dataset ---
    uspto_path = input("Enter USPTO PKL path: ").strip()
    with open(uspto_path, 'rb') as f:
        uspto_data = pickle.load(f)
    df = pd.DataFrame({"reaction": uspto_data})
    N = len(df)
    print(f"[INFO] Loaded USPTO dataset ({N} molecules)")

    # --- Phase 1: baseline model logits & value preds ---
    # placeholder: replace with actual model inference
    policy_logits = torch.randn(N, 5, device=device)
    value_preds = torch.randn(N, 1, device=device)

    # --- Phase 2: generate UQ files ---
    generate_uq_files(policy_logits, value_preds, save_dir='uq_outputs', required_len=N)

    # --- Phase 2: create weighted PKLs ---
    weighted_dir = 'uq_weighted_pkls'
    os.makedirs(weighted_dir, exist_ok=True)
    for uq_file in sorted(os.listdir('uq_outputs')):
        uq_path = os.path.join('uq_outputs', uq_file)
        weighted_df = reweight_dataset(df.copy(), uq_path)
        weighted_path = os.path.join(weighted_dir, uq_file.replace('.csv', '.pkl'))
        weighted_df.to_pickle(weighted_path)
        print(f"[INFO] Saved weighted PKL: {weighted_path}")

    # --- Phase 3: run MCTS + A* on weighted PKLs ---
    known_mols = set()  # load your known molecules from origin_dict.csv if needed
    model_path, value_model_path = '/path/to/policy_model.ckpt', '/path/to/value_model.pt'
    expand_fn = None  # TODO: load policy model
    value_model = None  # TODO: load value model

    for w_pkl in sorted(os.listdir(weighted_dir)):
        weighted_path = os.path.join(weighted_dir, w_pkl)
        with open(weighted_path, 'rb') as f:
            targets = [{'reaction': s, 'weight': w} for s, w in zip(pickle.load(f)['reaction'], pickle.load(f)['weight'])]
        base_tag = w_pkl.replace('.pkl','')
        success_rate, avg_depth, avg_exp, avg_time, results = play_meea(base_tag, targets, known_mols, value_model, expand_fn, device)
        out = {"success_rate": success_rate, "avg_depth": avg_depth, "avg_expansions": avg_exp, "avg_time": avg_time, "details": results}
        out_path = f'./phase3_stats_{base_tag}.pkl'
        with open(out_path, 'wb') as f: pickle.dump(out, f)
        print(f"\n[INFO] Saved Phase 3 stats: {out_path}")

if __name__ == "__main__":
    main()
