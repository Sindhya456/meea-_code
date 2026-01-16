#!/usr/bin/env python3
# ===========================================
# MEEA / MCTS + A* UQ-aware Full Pipeline
# ===========================================
import os
import sys
import time
import torch
import heapq
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import torch.nn.functional as F
from policyNet import MLPModel
from valueEnsemble import ValueEnsemble
import multiprocessing
import re

# ===========================================
# Phase 0: GPU setup
# ===========================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")

# ===========================================
# SMILES enumeration for TTA
# ===========================================
class SmilesEnumerator:
    def __init__(self, charset='@C)(=cOn1S2/H[N]\\', pad=120, leftpad=True, isomericSmiles=True, enum=True, canonical=False):
        self.charset = charset
        self.pad = pad
        self.leftpad = leftpad
        self.isomericSmiles = isomericSmiles
        self.enumerate = enum
        self.canonical = canonical
        self._char_to_int = {c:i for i,c in enumerate(charset)}
        self._int_to_char = {i:c for i,c in enumerate(charset)}

    def randomize_smiles(self, smiles):
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return smiles
        idxs = list(range(m.GetNumAtoms()))
        np.random.shuffle(idxs)
        nm = Chem.RenumberAtoms(m, idxs)
        return Chem.MolToSmiles(nm, canonical=self.canonical, isomericSmiles=self.isomericSmiles)

# ===========================================
# Policy / Value inference
# ===========================================
def prepare_expand(model_path):
    return MLPModel(model_path, 'template_rules.dat', device=device)

def prepare_value(model_f):
    model = ValueEnsemble(2048, 128, 0.1).to(device)
    ckpt = torch.load(model_f, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    return model

def smiles_to_fp(s, fp_dim=2048):
    mol = Chem.MolFromSmiles(s)
    if mol is None: return np.zeros(fp_dim, dtype=np.bool_)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool_)
    arr[list(fp.GetOnBits())] = 1
    return arr

def batch_smiles_to_fp(s_list, fp_dim=2048):
    return np.array([smiles_to_fp(s, fp_dim) for s in s_list])

def value_fn(model, mols):
    num_mols = len(mols)
    fps = batch_smiles_to_fp(mols).reshape(num_mols, -1)
    fps_tensor = torch.from_numpy(fps.astype(np.float32)).unsqueeze(0).to(device)
    mask_tensor = torch.ones_like(fps_tensor[:, :, 0])
    with torch.no_grad():
        v = model(fps_tensor, mask_tensor)
    return v.cpu().flatten()

# ===========================================
# Phase 1: Baseline MCTS + A*
# ===========================================
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

def meea_star(target_mol, known_mols, value_model, expand_fn, max_steps=500):
    target_smiles = target_mol['reaction'] if isinstance(target_mol, dict) else target_mol
    weight = float(target_mol.get('weight', 1.0)) if isinstance(target_mol, dict) else 1.0
    start_time = time.time()
    root_h = float(value_fn(value_model, [target_smiles]))
    root = AStarNode([target_smiles], g=0.0, h=root_h)
    open_list, visited = [root], set()
    expansions = 0

    while open_list and expansions < max_steps:
        node = heapq.heappop(open_list)
        expansions += 1
        if all(m in known_mols for m in node.state):
            return True, node, expansions, time.time()-start_time

        expanded_mol = node.state[0]
        expanded_policy = expand_fn.run(expanded_mol, topk=50)
        if expanded_policy is None: continue
        scores = expanded_policy.get('scores', [])
        reactant_list = expanded_policy.get('reactants', [])
        templates = expanded_policy.get('template', [])

        for i in range(len(scores)):
            raw_reactants = reactant_list[i]
            reactants = [r for r in raw_reactants.split('.') if r not in known_mols and r != '']
            new_state = sorted(list(set(reactants + node.state[1:])))
            state_key = '.'.join(new_state)
            if state_key in visited: continue
            visited.add(state_key)
            cost = -np.log(np.clip(scores[i], 1e-6, 1.0)) * (2.0 - weight)
            h = float(value_fn(value_model, new_state)) if new_state else 0.0
            child = AStarNode(new_state, g=node.g+cost, h=h, parent=node,
                              action=(templates[i] if i < len(templates) else None, raw_reactants))
            heapq.heappush(open_list, child)
    return False, None, expansions, time.time()-start_time

def play_meea(mols, known_mols, value_model, expand_fn):
    results = []
    for mol in mols:
        success, node, exp, elapsed = meea_star(mol, known_mols, value_model, expand_fn)
        depth = node.g if success else 32
        results.append({"success": success, "depth": depth, "expansions": exp, "time": elapsed})
    return results

# ===========================================
# Phase 2: UQ Generation
# ===========================================
def compute_aleatoric(policy_logits_tta):
    probs = F.softmax(policy_logits_tta, dim=1)
    mean_probs = probs.mean(dim=0)
    entropy = -(mean_probs*torch.log(mean_probs+1e-12)).sum()
    return entropy.item(), mean_probs

def compute_epistemic_jsd(policy_logits, value_preds):
    policy_probs = F.softmax(policy_logits, dim=1)
    value_probs = torch.sigmoid(value_preds).squeeze(-1)
    value_probs = torch.stack([value_probs, 1-value_probs], dim=1)
    if policy_probs.shape[1] != 2:
        top2, _ = torch.topk(policy_probs, 2, dim=1)
        policy_probs = top2 / top2.sum(dim=1, keepdim=True)
    M = 0.5*(policy_probs+value_probs)
    jsd = 0.5*((policy_probs*(torch.log(policy_probs+1e-12)-torch.log(M+1e-12))).sum(dim=1) +
               (value_probs*(torch.log(value_probs+1e-12)-torch.log(M+1e-12))).sum(dim=1))
    return jsd.cpu().numpy()

def uq_analysis(policy_logits, value_preds, alea_weight=0.5, epis_weight=0.5):
    B, C = policy_logits.shape
    alea = [compute_aleatoric(policy_logits[i].unsqueeze(0))[0] for i in range(B)]
    epis = compute_epistemic_jsd(policy_logits, value_preds)
    combined = alea_weight*np.array(alea)+epis_weight*np.array(epis)
    return combined

def generate_uq_csvs(policy_logits, value_preds, dataset_len):
    os.makedirs("uq_outputs", exist_ok=True)
    weights = [(round(e,1), round(1-e,1)) for e in np.linspace(0.1,0.9,9)]
    # Pad tensors if needed
    if policy_logits.shape[0] < dataset_len:
        pad = dataset_len - policy_logits.shape[0]
        policy_logits = torch.cat([policy_logits, torch.zeros(pad, policy_logits.shape[1], device=device)], dim=0)
        value_preds = torch.cat([value_preds, torch.zeros(pad,1, device=device)], dim=0)
    for alea, epis in weights:
        uq_scores = uq_analysis(policy_logits, value_preds, alea_weight=alea, epis_weight=epis)
        fname = f"uq_alea{alea:.1f}_epis{epis:.1f}.csv"
        pd.DataFrame({"uq_score": uq_scores}).to_csv(os.path.join("uq_outputs", fname), index=False)
        print(f"✅ Saved UQ CSV: {fname}")

# ===========================================
# Phase 2b: Exponential reweighting
# ===========================================
def reweight_exponential(uq_scores, gamma=2.0, normalize=True):
    uq_scores = torch.tensor(uq_scores, dtype=torch.float32, device=device)
    weights = torch.exp(-gamma*uq_scores)
    if normalize:
        weights /= weights.sum()
    return weights.cpu().numpy()

def assign_weights(dataset, uq_scores):
    weights = reweight_exponential(uq_scores)
    for i, d in enumerate(dataset):
        d['weight'] = float(weights[i])
    return dataset

# ===========================================
# Utilities
# ===========================================
def normalize_targets(obj):
    if isinstance(obj, pd.DataFrame):
        if 'reaction' in obj.columns:
            records = obj.to_dict(orient='records')
        elif 'mol' in obj.columns:
            records = [{'reaction': s} for s in obj['mol']]
        else:
            records = obj.to_dict(orient='records')
    elif isinstance(obj, list):
        records = [{'reaction': x} if not isinstance(x, dict) else x for x in obj]
    else:
        raise TypeError("Unsupported type")
    return records

def prepare_starting_mols(path='origin_dict.csv'):
    df = pd.read_csv(path)
    return set(df['mol'].astype(str).tolist())

# ===========================================
# Main Pipeline
# ===========================================
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    # Paths
    policy_model_path = "policy_model.ckpt"
    value_model_path = "value_pc.pt"
    dataset_path = input("Enter weighted or raw PKL dataset path: ").strip()

    # Load dataset
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    targets = normalize_targets(dataset)
    dataset_len = len(targets)
    print(f"[INFO] Loaded dataset with {dataset_len} molecules")

    # Models
    expand_fn = prepare_expand(policy_model_path)
    value_model = prepare_value(value_model_path)

    known_mols = prepare_starting_mols()

    # -----------------------------
    # Phase 1: Baseline MCTS/A*
    # -----------------------------
    print("[PHASE 1] Running baseline MCTS + A*...")
    results_phase1 = play_meea(targets, known_mols, value_model, expand_fn)
    print("[PHASE 1] Done")

    # Collect policy logits & value preds for UQ
    # For demonstration, using random tensors; replace with network output
    policy_logits = torch.randn(dataset_len, 5, device=device)
    value_preds = torch.randn(dataset_len, 1, device=device)

    # -----------------------------
    # Phase 2: Generate UQ CSVs
    # -----------------------------
    print("[PHASE 2] Generating UQ CSVs...")
    generate_uq_csvs(policy_logits, value_preds, dataset_len)
    print("[PHASE 2] Done")

    # Example: use first CSV to reweight
    uq_example_csv = "uq_outputs/uq_alea0.9_epis0.1.csv"
    uq_scores = pd.read_csv(uq_example_csv)['uq_score'].to_numpy()
    targets_weighted = assign_weights(targets, uq_scores)

    # -----------------------------
    # Phase 3: Weighted MCTS + A*
    # -----------------------------
    print("[PHASE 3] Running weighted MCTS + A*...")
    results_phase3 = play_meea(targets_weighted, known_mols, value_model, expand_fn)
    print("[PHASE 3] Done")

    print("\n✅ Full pipeline finished successfully!")
