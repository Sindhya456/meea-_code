#!/usr/bin/env python3
# ====================================================
# Multi-GPU MCTS + A* + UQ-aware pipeline (TXT output)
# ====================================================

import os, pickle, torch, numpy as np, pandas as pd
from tqdm import tqdm
from policyNet import MLPModel
from valueEnsemble import ValueEnsemble

# ---------------------------
# GPU setup
# ---------------------------
gpus = list(range(torch.cuda.device_count()))
device_names = [f'cuda:{i}' for i in gpus]
print(f"[INFO] Detected GPUs: {device_names}")

# ---------------------------
# Prepare test folder
# ---------------------------
test_dir = './test'
os.makedirs(test_dir, exist_ok=True)
# Clean previous files
for f in os.listdir(test_dir):
    os.remove(os.path.join(test_dir, f))
print(f"[INFO] Cleaned previous test results in {test_dir}")

# ---------------------------
# Load models
# ---------------------------
policy_model_path = './saved_model/policy_model.ckpt'
value_model_path = './saved_model/value_pc.pt'
template_rules_path = './saved_model/template_rules.dat'

expand_models = [MLPModel(policy_model_path, template_rules_path, device=d) for d in device_names]
value_models = []
for d in device_names:
    model = ValueEnsemble(2048, 128, 0.1).to(d)
    model.load_state_dict(torch.load(value_model_path, map_location=d))
    model.eval()
    value_models.append(model)

# ---------------------------
# Fingerprints
# ---------------------------
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_fp(s, fp_dim=2048):
    mol = Chem.MolFromSmiles(s)
    if mol is None: return np.zeros(fp_dim, dtype=np.bool_)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool_)
    arr[list(fp.GetOnBits())] = 1
    return arr

def batch_smiles_to_fp(s_list):
    return np.array([smiles_to_fp(s) for s in s_list])

# ---------------------------
# Value function
# ---------------------------
def value_fn(model, mols, device):
    fps = batch_smiles_to_fp(mols).astype(np.float32)
    fps_tensor = torch.from_numpy(fps).unsqueeze(0).to(device)
    mask_tensor = torch.ones(1, fps_tensor.shape[1], dtype=torch.float32).to(device)
    with torch.no_grad():
        v = model(fps_tensor, mask_tensor).cpu().numpy()
    return v.flatten()

# ---------------------------
# MCTS + A* node
# ---------------------------
import heapq
class AStarNode:
    def __init__(self, state, g, h, action=None, parent=None):
        self.state = state
        self.g = g
        self.h = h
        self.f = g+h
        self.action = action
        self.parent = parent
    def __lt__(self, other):
        return self.f < other.f

def meea_star(target, known_mols, value_model, expand_fn, device, max_steps=500):
    import numpy as np
    import time
    start = time.time()
    h0 = value_fn(value_model, [target], device)[0]
    root = AStarNode([target], g=0, h=h0)
    open_list = [root]
    visited = set()
    expansions = 0
    while open_list and expansions<max_steps:
        node = heapq.heappop(open_list)
        expansions +=1
        if all(m in known_mols for m in node.state):
            return True, node, expansions, time.time()-start
        mol = node.state[0]
        policy_out = expand_fn.run(mol, topk=50)
        if policy_out is None: continue
        for i in range(len(policy_out['scores'])):
            reactants = [r for r in policy_out['reactants'][i].split('.') if r not in known_mols]
            reactants = sorted(list(set(reactants+node.state[1:])))
            key = '.'.join(reactants)
            if key in visited: continue
            visited.add(key)
            cost = -np.log(np.clip(policy_out['scores'][i], 1e-3, 1.0))
            h = value_fn(value_model, reactants, device)[0] if reactants else 0
            child = AStarNode(reactants, g=node.g+cost, h=h, parent=node, action=(policy_out['template'][i], policy_out['reactants'][i]))
            heapq.heappush(open_list, child)
    return False, None, expansions, time.time()-start

# ---------------------------
# UQ functions
# ---------------------------
def compute_aleatoric(policy_logits):
    import torch
    probs = torch.softmax(policy_logits, dim=1)
    mean_probs = probs.mean(dim=0)
    return -(mean_probs*torch.log(mean_probs+1e-12)).sum().item()

def compute_epistemic(policy_logits, value_preds):
    import torch
    policy_probs = torch.softmax(policy_logits, dim=1)
    value_probs = torch.sigmoid(value_preds).squeeze(-1)
    value_probs = torch.stack([value_probs,1-value_probs],dim=1)
    if policy_probs.shape[1]!=2:
        top2,_=torch.topk(policy_probs,2,dim=1)
        policy_probs=top2/top2.sum(1,keepdim=True)
    M=0.5*(policy_probs+value_probs)
    jsd=0.5*((policy_probs*(torch.log(policy_probs+1e-12)-torch.log(M+1e-12))).sum(1)+
             (value_probs*(torch.log(value_probs+1e-12)-torch.log(M+1e-12)).sum(1)))
    return jsd.cpu().numpy()

def compute_combined_uq(alea, epis, w_alea=0.5, w_epis=0.5):
    return w_alea*np.array(alea).ravel()+w_epis*np.array(epis).ravel()

# ---------------------------
# Weighted dataset
# ---------------------------
def create_weighted_dataset(mols, uq_scores, method='exponential', scale=5.0):
    uq_scores = np.array(uq_scores).ravel()
    if method=='linear':
        weights = uq_scores/uq_scores.sum()
    elif method=='exponential':
        exp_scores = np.exp(scale*uq_scores)
        weights = exp_scores/exp_scores.sum()
    else: raise ValueError(f"Unknown weighting method: {method}")
    indices = np.random.choice(len(mols), size=len(mols), p=weights)
    weighted_mols = [mols[i] for i in indices]
    return weighted_mols, weights

# ---------------------------
# Main pipeline
# ---------------------------
if __name__=='__main__':
    import math

    # Load datasets
    dataset_files = os.listdir('./test_dataset')
    for ds_file in dataset_files:
        ds_path = os.path.join('./test_dataset', ds_file)
        with open(ds_path,'rb') as f:
            mols = pickle.load(f)
        ds_name = os.path.splitext(ds_file)[0]
        print(f"[INFO] Dataset: {ds_name}, molecules: {len(mols)}")

        # Split molecules across GPUs
        num_gpus = len(gpus)
        mol_chunks = np.array_split(mols, num_gpus)

        # Phase 1: Baseline MCTS
        baseline_results = []
        for gpu_idx, chunk in enumerate(mol_chunks):
            value_model = value_models[gpu_idx]
            expand_fn = expand_models[gpu_idx]
            device = device_names[gpu_idx]
            for mol in tqdm(chunk, desc=f"Phase1 MCTS ({ds_name}) GPU{gpu_idx}"):
                success,node,exp_count,elapsed = meea_star(mol, mols, value_model, expand_fn, device)
                depth=node.g if success else 32
                baseline_results.append({"success":success,"depth":depth,"expansions":exp_count,"time":elapsed})

        # Save Phase 1 results as TXT
        txt_file = os.path.join(test_dir,f'stat_baseline_{ds_name}.txt')
        with open(txt_file,'w') as f:
            f.write(f"Dataset: {ds_name}\n")
            f.write(f"Success rate: {np.mean([r['success'] for r in baseline_results]):.3f}\n")
            f.write(f"Avg depth: {np.mean([r['depth'] for r in baseline_results]):.2f}\n")
            f.write(f"Avg expansions: {np.mean([r['expansions'] for r in baseline_results]):.2f}\n")
            f.write(f"Avg time: {np.mean([r['time'] for r in baseline_results]):.2f}s\n")
        print(f"[INFO] Phase 1 results saved: {txt_file}")

        # Phase 2: UQ
        policy_logits = []
        value_preds = []
        for gpu_idx, chunk in enumerate(mol_chunks):
            value_model = value_models[gpu_idx]
            expand_fn = expand_models[gpu_idx]
            device = device_names[gpu_idx]
            for mol in tqdm(chunk, desc=f"Phase2 UQ ({ds_name}) GPU{gpu_idx}"):
                po = expand_fn.run(mol, topk=50)
                if po is not None:
                    logits = torch.tensor(po['scores']).unsqueeze(0)
                    policy_logits.append(logits)
                    vpred = value_fn(value_model, [mol], device)
                    value_preds.append(torch.tensor([[vpred]]))
        policy_logits = torch.cat(policy_logits, dim=0).to(device_names[0])
        value_preds = torch.cat(value_preds, dim=0).to(device_names[0])

        # Compute combined UQ
        alea = [compute_aleatoric(pl.unsqueeze(0)) for pl in policy_logits]
        epis = compute_epistemic(policy_logits, value_preds)
        combined_uq = compute_combined_uq(alea, epis, w_alea=0.5, w_epis=0.5)

        # Save UQ scores
        uq_file = os.path.join('./uq_outputs',f'uq_combined_{ds_name}.txt')
        os.makedirs('./uq_outputs', exist_ok=True)
        with open(uq_file,'w') as f:
            for score in combined_uq:
                f.write(f"{score:.6f}\n")
        print(f"[INFO] UQ scores saved: {uq_file}")

        # Phase 3: Weighted MCTS (UQ-aware)
        weighted_mols, _ = create_weighted_dataset(mols, combined_uq)
        weighted_results = []
        for gpu_idx, chunk in enumerate(np.array_split(weighted_mols,num_gpus)):
            value_model = value_models[gpu_idx]
            expand_fn = expand_models[gpu_idx]
            device = device_names[gpu_idx]
            for mol in tqdm(chunk, desc=f"Phase3 Weighted MCTS ({ds_name}) GPU{gpu_idx}"):
                success,node,exp_count,elapsed = meea_star(mol, mols, value_model, expand_fn, device)
                depth=node.g if success else 32
                weighted_results.append({"success":success,"depth":depth,"expansions":exp_count,"time":elapsed})

        # Save Phase 3
        txt_file = os.path.join(test_dir,f'stat_meea_{ds_name}.txt')
        with open(txt_file,'w') as f:
            f.write(f"Dataset: {ds_name}\n")
            f.write(f"Success rate: {np.mean([r['success'] for r in weighted_results]):.3f}\n")
            f.write(f"Avg depth: {np.mean([r['depth'] for r in weighted_results]):.2f}\n")
            f.write(f"Avg expansions: {np.mean([r['expansions'] for r in weighted_results]):.2f}\n")
            f.write(f"Avg time: {np.mean([r['time'] for r in weighted_results]):.2f}s\n")
        print(f"[INFO] Phase 3 results saved: {txt_file}")

    print("[INFO] Pipeline complete for all datasets!")
