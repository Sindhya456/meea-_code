#!/usr/bin/env python3
# ====================================================
# Full MCTS + A* + UQ-aware pipeline (GPU-ready)
# Multi-GPU safe, with average statistics per dataset
# ====================================================

import os, pickle, time, heapq
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process
from rdkit import Chem
from rdkit.Chem import AllChem

from valueEnsemble import ValueEnsemble
from policyNet import MLPModel

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
    fps_tensor = torch.from_numpy(fps).to(device)
    mask_tensor = torch.ones(fps_tensor.shape, dtype=torch.float32).to(device)
    with torch.no_grad():
        v = model(fps_tensor.unsqueeze(0), mask_tensor.unsqueeze(0)).cpu().numpy()
    return v.flatten()

# ---------------------------
# Models
# ---------------------------
def prepare_expand(model_path, device='cuda'):
    return MLPModel(model_path, './saved_model/template_rules.dat', device=device)

def prepare_value(model_path, device='cuda'):
    model = ValueEnsemble(2048, 128, 0.1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# ---------------------------
# Phase 1 & 3: A* Node
# ---------------------------
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
    start = time.time()
    h0 = value_fn(value_model, [target], device)[0]
    root = AStarNode([target], g=0, h=h0)
    open_list = [root]
    visited = set()
    expansions = 0

    while open_list and expansions < max_steps:
        node = heapq.heappop(open_list)
        expansions +=1
        if all(m in known_mols for m in node.state):
            return True, node, expansions, time.time()-start
        # Expand node
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
            child = AStarNode(reactants, g=node.g+cost, h=h, parent=node,
                              action=(policy_out['template'][i], policy_out['reactants'][i]))
            heapq.heappush(open_list, child)
    return False, None, expansions, time.time()-start

# ---------------------------
# Phase 2: UQ
# ---------------------------
def compute_aleatoric(policy_logits):
    probs = torch.softmax(policy_logits, dim=1)
    mean_probs = probs.mean(dim=0)
    return -(mean_probs*torch.log(mean_probs+1e-12)).sum().item()

def compute_epistemic(policy_logits, value_preds):
    policy_probs = torch.softmax(policy_logits, dim=1)
    value_probs = torch.sigmoid(value_preds).unsqueeze(1)
    value_probs = torch.cat([value_probs, 1-value_probs], dim=1)
    if policy_probs.shape[1]!=2:
        top2,_=torch.topk(policy_probs,2,dim=1)
        policy_probs=top2/top2.sum(1,keepdim=True)
    M=0.5*(policy_probs+value_probs)
    jsd=0.5*((policy_probs*(torch.log(policy_probs+1e-12)-torch.log(M+1e-12))).sum(1)+
             (value_probs*(torch.log(value_probs+1e-12)-torch.log(M+1e-12))).sum(1))
    return jsd.cpu().numpy()

def compute_combined_uq(alea, epis, w_alea=0.5, w_epis=0.5):
    return w_alea*np.array(alea).ravel()+w_epis*np.array(epis).ravel()

def generate_uq(policy_logits, value_preds, save_dir="./uq_outputs", w_alea=0.5, w_epis=0.5):
    os.makedirs(save_dir, exist_ok=True)
    alea = [compute_aleatoric(pl.unsqueeze(0)) for pl in policy_logits]
    epis = compute_epistemic(policy_logits, value_preds)
    combined = compute_combined_uq(alea, epis, w_alea=w_alea, w_epis=w_epis)
    uq_file = os.path.join(save_dir, f"uq_alea{w_alea}_epis{w_epis}.txt")
    np.savetxt(uq_file, combined)
    return uq_file, combined

# ---------------------------
# Weighted sampling
# ---------------------------
def create_weighted_dataset(mols, uq_scores, method='exponential', scale=5.0):
    uq_scores = np.array(uq_scores).ravel()
    if method=='linear':
        weights = uq_scores/uq_scores.sum()
    elif method=='exponential':
        exp_scores = np.exp(scale*uq_scores)
        weights = exp_scores/exp_scores.sum()
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    indices = np.random.choice(len(mols), size=len(mols), p=weights)
    weighted_mols = [mols[i] for i in indices]
    return weighted_mols

# ---------------------------
# Main
# ---------------------------
if __name__=='__main__':
    os.makedirs('./test', exist_ok=True)
    os.makedirs('./uq_outputs', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load models
    expand_fn = prepare_expand('./saved_model/policy_model.ckpt', device=device)
    value_model = prepare_value('./saved_model/value_pc.pt', device=device)

    # Datasets
    datasets = os.listdir('./test_dataset')
    for ds_file in datasets:
        ds_path = os.path.join('./test_dataset', ds_file)
        with open(ds_path,'rb') as f:
            mols = pickle.load(f)
        ds_name = os.path.splitext(ds_file)[0]
        known_mols = set(mols)

        print(f"[INFO] Dataset: {ds_name}, molecules: {len(mols)}")

        # -------- Phase 1: Baseline MCTS + A* --------
        baseline_results = []
        for mol in tqdm(mols, desc=f"Phase1 MCTS+A* ({ds_name})"):
            success,node,exp_count,elapsed = meea_star(mol, known_mols, value_model, expand_fn, device)
            depth = node.g if success else 32
            baseline_results.append({"success":success,"depth":depth,"expansions":exp_count,"time":elapsed})

        # Save Phase 1 stats (averages)
        out_path = f'./test/stat_baseline_{ds_name}.txt'
        with open(out_path,'w') as f:
            success_rate = np.mean([r["success"] for r in baseline_results])
            avg_depth = np.mean([r["depth"] for r in baseline_results if r["success"]]) if any(r["success"] for r in baseline_results) else 0
            avg_exp = np.mean([r["expansions"] for r in baseline_results])
            avg_time = np.mean([r["time"] for r in baseline_results])
            f.write(f"Dataset: {ds_name}\nSuccess rate: {success_rate:.3f}\nAvg depth: {avg_depth:.2f}\nAvg expansions: {avg_exp:.2f}\nAvg time: {avg_time:.2f}s\n")
        print(f"[INFO] Phase 1 results saved: {out_path}")

        # -------- Phase 2: Generate policy logits & value preds --------
        policy_logits = torch.stack([expand_fn.run(m) for m in mols]).to(device)
        value_preds = torch.tensor([value_fn(value_model,[m],device)[0] for m in mols]).to(device)

        uq_file, uq_scores = generate_uq(policy_logits, value_preds, save_dir='./uq_outputs', w_alea=0.5, w_epis=0.5)
        print(f"[INFO] UQ scores saved: {uq_file}")

        # -------- Phase 3: Weighted MCTS + A* --------
        weighted_mols = create_weighted_dataset(mols, uq_scores)
        weighted_results = []
        for mol in tqdm(weighted_mols, desc=f"Phase3 Weighted MCTS+A* ({ds_name})"):
            success,node,exp_count,elapsed = meea_star(mol, known_mols, value_model, expand_fn, device)
            depth = node.g if success else 32
            weighted_results.append({"success":success,"depth":depth,"expansions":exp_count,"time":elapsed})

        # Save Phase 3 stats (averages)
        out_path = f'./test/stat_meea_{ds_name}.txt'
        with open(out_path,'w') as f:
            success_rate = np.mean([r["success"] for r in weighted_results])
            avg_depth = np.mean([r["depth"] for r in weighted_results if r["success"]]) if any(r["success"] for r in weighted_results) else 0
            avg_exp = np.mean([r["expansions"] for r in weighted_results])
            avg_time = np.mean([r["time"] for r in weighted_results])
            f.write(f"Dataset: {ds_name}\nSuccess rate: {success_rate:.3f}\nAvg depth: {avg_depth:.2f}\nAvg expansions: {avg_exp:.2f}\nAvg time: {avg_time:.2f}s\n")
        print(f"[INFO] Phase 3 results saved: {out_path}")

    print("[INFO] Pipeline complete for all datasets!")
