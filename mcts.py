#!/usr/bin/env python3
# ====================================================
# Full MCTS + A* + UQ-aware pipeline (GPU-ready, multi-GPU safe)
# Loops over all datasets in test_dataset
# Results saved as TXT for easy shell access
# ====================================================

import os, pickle, time
import torch, numpy as np, pandas as pd
import heapq
from tqdm import tqdm

# ---------------------------
# Models & Helpers
# ---------------------------
from valueEnsemble import ValueEnsemble
from policyNet import MLPModel
from rdkit import Chem
from rdkit.Chem import AllChem

# ---------------------------
# Smiles Enumerator for TTA
# ---------------------------
class SmilesEnumerator:
    def __init__(self, charset='@C)(=cOn1S2/H[N]\\', pad=120, leftpad=True, isomericSmiles=True, enum=True, canonical=False):
        self._charset = charset
        self.pad = pad
        self.leftpad = leftpad
        self.isomericSmiles = isomericSmiles
        self.enumerate = enum
        self.canonical = canonical

    def randomize_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return smiles
        idxs = list(range(mol.GetNumAtoms()))
        np.random.shuffle(idxs)
        mol2 = Chem.RenumberAtoms(mol, idxs)
        return Chem.MolToSmiles(mol2, canonical=self.canonical, isomericSmiles=self.isomericSmiles)

# ---------------------------
# Fingerprints
# ---------------------------
def smiles_to_fp(s, fp_dim=2048):
    mol = Chem.MolFromSmiles(s)
    if mol is None: return np.zeros(fp_dim, dtype=np.bool_)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool_)
    arr[list(fp.GetOnBits())] = 1
    return arr

def batch_smiles_to_fp(s_list, fp_dim=2048):
    return np.array([smiles_to_fp(s, fp_dim) for s in s_list])

def value_fn(model, mols, device='cuda'):
    fps = batch_smiles_to_fp(mols).astype(np.float32)
    fps_tensor = torch.from_numpy(fps).unsqueeze(0).to(device)
    mask_tensor = torch.ones(1, fps_tensor.shape[1], dtype=torch.float32).to(device)
    with torch.no_grad():
        v = model(fps_tensor, mask_tensor).cpu().numpy()
    return v.flatten()

# ---------------------------
# Prepare models
# ---------------------------
def prepare_expand(model_path, device='cuda'):
    return MLPModel(model_path, './saved_model/template_rules.dat', device=device)

def prepare_value(model_f, device='cuda'):
    model = ValueEnsemble(2048, 128, 0.1).to(device)
    model.load_state_dict(torch.load(model_f, map_location=device))
    model.eval()
    return model

# ---------------------------
# Phase 1 & 3: MCTS + A* search
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

def meea_star(target, known_mols, value_model, expand_fn, device, max_steps=500):
    value_model = value_model.to(device)
    start = time.time()
    h0 = value_fn(value_model, [target], device)[0]
    root = AStarNode([target], g=0, h=h0)
    open_list = [root]
    visited = set()
    expansions = 0

    while open_list and expansions < max_steps:
        node = heapq.heappop(open_list)
        expansions += 1
        if all(m in known_mols for m in node.state):
            return True, node, expansions, time.time() - start
        # Expand
        mol = node.state[0]
        policy_out = expand_fn.run(mol, topk=50)
        if policy_out is None: continue
        for i in range(len(policy_out['scores'])):
            reactants = [r for r in policy_out['reactants'][i].split('.') if r not in known_mols]
            reactants = sorted(list(set(reactants + node.state[1:])))
            key = '.'.join(reactants)
            if key in visited: continue
            visited.add(key)
            cost = -np.log(np.clip(policy_out['scores'][i], 1e-3, 1.0))
            h = value_fn(value_model, reactants, device)[0] if reactants else 0
            child = AStarNode(reactants, g=node.g + cost, h=h, parent=node,
                              action=(policy_out['template'][i], policy_out['reactants'][i]))
            heapq.heappush(open_list, child)
    return False, None, expansions, time.time() - start

# ---------------------------
# Phase 2: UQ calculation
# ---------------------------
def compute_aleatoric(policy_logits):
    probs = torch.softmax(policy_logits, dim=1)
    mean_probs = probs.mean(dim=0)
    return -(mean_probs * torch.log(mean_probs + 1e-12)).sum().item()

def compute_epistemic(policy_logits, value_preds):
    policy_probs = torch.softmax(policy_logits, dim=1)
    value_probs = torch.sigmoid(value_preds).squeeze(-1)
    value_probs = torch.stack([value_probs, 1 - value_probs], dim=1)
    if policy_probs.shape[1] != 2:
        top2, _ = torch.topk(policy_probs, 2, dim=1)
        policy_probs = top2 / top2.sum(1, keepdim=True)
    M = 0.5 * (policy_probs + value_probs)
    jsd = 0.5 * ((policy_probs * (torch.log(policy_probs + 1e-12) - torch.log(M + 1e-12))).sum(1) +
                 (value_probs * (torch.log(value_probs + 1e-12) - torch.log(M + 1e-12))).sum(1))
    return jsd.cpu().numpy()

def compute_combined_uq(alea, epis, w_alea=0.5, w_epis=0.5):
    return w_alea*np.array(alea).ravel() + w_epis*np.array(epis).ravel()

def generate_uq_files(policy_logits, value_preds, save_dir="uq_outputs", required_len=None):
    os.makedirs(save_dir, exist_ok=True)
    weights = [(round(e,1), round(1-e,1)) for e in np.linspace(0.1,0.9,9)]
    if required_len is None: required_len = len(policy_logits)
    if len(policy_logits) != required_len:
        raise ValueError(f"Expected {required_len} samples, got {len(policy_logits)}")
    for epis_w, alea_w in weights:
        alea=[]
        for i in range(len(policy_logits)):
            ent = compute_aleatoric(policy_logits[i].unsqueeze(0))
            alea.append(ent)
        epis = compute_epistemic(policy_logits, value_preds)
        combined = compute_combined_uq(alea, epis, w_alea=alea_w, w_epis=epis_w)
        filename = f"uq_alea{alea_w:.1f}_epis{epis_w:.1f}.txt"
        with open(os.path.join(save_dir, filename),'w') as f:
            f.write("\n".join([f"{x:.6f}" for x in combined]))

# ---------------------------
# Phase 2b: Weighted dataset
# ---------------------------
def create_weighted_dataset(mols, uq_scores, method='exponential', scale=5.0):
    uq_scores = np.array(uq_scores).ravel()
    if method=='linear':
        weights = uq_scores / uq_scores.sum()
    elif method=='exponential':
        exp_scores = np.exp(scale*uq_scores)
        weights = exp_scores / exp_scores.sum()
    else: raise ValueError(f"Unknown weighting method: {method}")
    indices = np.random.choice(len(mols), size=len(mols), p=weights)
    weighted_mols = [mols[i] for i in indices]
    return weighted_mols, weights

# ---------------------------
# Main pipeline
# ---------------------------
if __name__=='__main__':
    os.makedirs('./test', exist_ok=True)
    os.makedirs('./uq_outputs', exist_ok=True)
    device_names = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ['cpu']

    # Load models once
    expand_fn = prepare_expand('./saved_model/policy_model.ckpt', device=device_names[0])
    value_model = prepare_value('./saved_model/value_pc.pt', device=device_names[0])

    # Loop over datasets
    datasets = os.listdir('./test_dataset')
    for ds_file in datasets:
        ds_path = os.path.join('./test_dataset', ds_file)
        with open(ds_path,'rb') as f: mols = pickle.load(f)
        ds_name = os.path.splitext(ds_file)[0]
        known_mols = set(mols)

        print(f"[INFO] Dataset: {ds_name}, molecules: {len(mols)}")

        # ----------------- Phase 1 -----------------
        baseline_results=[]
        for i, mol in enumerate(tqdm(mols, desc=f"Phase1 MCTS+A* ({ds_name})")):
            dev = device_names[i % len(device_names)]
            success, node, exp_count, elapsed = meea_star(mol, known_mols, value_model, expand_fn, dev)
            depth = node.g if success else 32
            baseline_results.append({"success":success,"depth":depth,"expansions":exp_count,"time":elapsed})

        # Save baseline
        out_path = f'./test/stat_baseline_{ds_name}.txt'
        with open(out_path,'w') as f:
            for r in baseline_results:
                f.write(f"{r['success']}\t{r['depth']:.2f}\t{r['expansions']}\t{r['time']:.4f}\n")
        print(f"[INFO] Phase 1 results saved: {out_path}")

        # ----------------- Phase 2 -----------------
        # Generate policy logits safely
        policy_logits_list = []
        for i, mol in enumerate(tqdm(mols, desc=f"Generating policy logits ({ds_name})")):
            dev = device_names[i % len(device_names)]
            expand_fn.model = expand_fn.model.to(dev)
            out = expand_fn.run(mol)
            if out is None or len(out['scores']) == 0:
                scores = torch.zeros(5, device=dev)
            else:
                scores = torch.tensor(out['scores'][:5], device=dev)
                if len(scores) < 5:
                    scores = torch.cat([scores, torch.zeros(5-len(scores), device=dev)])
            policy_logits_list.append(scores)
        policy_logits = torch.stack(policy_logits_list).to(device_names[0])

        # Value predictions
        value_preds = torch.tensor([value_fn(value_model, [m], device_names[0])[0] for m in mols], device=device_names[0])

        generate_uq_files(policy_logits, value_preds, save_dir='./uq_outputs', required_len=len(mols))

        # ----------------- Phase 2b -----------------
        uq_file = './uq_outputs/uq_alea0.5_epis0.5.txt'
        uq_scores = np.loadtxt(uq_file)
        weighted_mols, weights = create_weighted_dataset(mols, uq_scores)

        # ----------------- Phase 3 -----------------
        weighted_results=[]
        for i, mol in enumerate(tqdm(weighted_mols, desc=f"Phase3 Weighted MCTS+A* ({ds_name})")):
            dev = device_names[i % len(device_names)]
            success, node, exp_count, elapsed = meea_star(mol, known_mols, value_model, expand_fn, dev)
            depth = node.g if success else 32
            weighted_results.append({"success":success,"depth":depth,"expansions":exp_count,"time":elapsed})

        # Save Phase 3
        out_path = f'./test/stat_meea_{ds_name}.txt'
        with open(out_path,'w') as f:
            for r in weighted_results:
                f.write(f"{r['success']}\t{r['depth']:.2f}\t{r['expansions']}\t{r['time']:.4f}\n")
        print(f"[INFO] Phase 3 results saved: {out_path}")

    print("[INFO] Pipeline complete for all datasets!")
