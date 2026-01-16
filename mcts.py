#!/usr/bin/env python3
# full_pipeline_gpu_loop.py
# Unified MCTS + A* + UQ pipeline, GPU-aware, loops over all datasets in test_datasets/

import os, sys, pickle, time, heapq, re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from policyNet import MLPModel
from valueEnsemble import ValueEnsemble

# =======================================
# GPU Setup
# =======================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# =======================================
# -------------------------
# SmilesEnumerator
# -------------------------
class SmilesEnumerator:
    def __init__(self, charset='@C)(=cOn1S2/H[N]\\', pad=120, leftpad=True,
                 isomericSmiles=True, enum=True, canonical=False):
        self._charset = charset
        self.pad = pad
        self.leftpad = leftpad
        self.isomericSmiles = isomericSmiles
        self.enumerate = enum
        self.canonical = canonical
        self._char_to_int = {c:i for i,c in enumerate(charset)}
        self._int_to_char = {i:c for i,c in enumerate(charset)}

    def randomize_smiles(self, smiles):
        m = Chem.MolFromSmiles(smiles)
        if m is None: return smiles
        idxs = list(range(m.GetNumAtoms()))
        np.random.shuffle(idxs)
        nm = Chem.RenumberAtoms(m, idxs)
        return Chem.MolToSmiles(nm, canonical=self.canonical, isomericSmiles=self.isomericSmiles)

# =======================================
# -------------------------
# Fingerprints
# -------------------------
def smiles_to_fp(s, fp_dim=2048):
    if isinstance(s, dict): s = s.get('reaction', '')
    s = str(s).strip()
    mol = Chem.MolFromSmiles(s)
    if mol is None: return np.zeros(fp_dim, dtype=np.bool_)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool_)
    arr[list(fp.GetOnBits())] = 1
    return arr

def batch_smiles_to_fp(s_list, fp_dim=2048):
    return np.array([smiles_to_fp(s, fp_dim) for s in s_list])

# =======================================
# -------------------------
# Models
# -------------------------
def prepare_expand(model_path):
    return MLPModel(model_path, 'template_rules.dat', device=str(device))

def prepare_value(model_f):
    model = ValueEnsemble(2048, 128, 0.1).to(device)
    ckpt = torch.load(model_f, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    return model

def value_fn(model, mols):
    fps = batch_smiles_to_fp(mols, fp_dim=2048).astype(np.float32)
    mask = np.ones(len(fps), dtype=np.float32)
    fps_tensor = torch.tensor(fps, dtype=torch.float32).unsqueeze(0).to(device)
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        v = model(fps_tensor, mask_tensor)
    return float(v.cpu().data.numpy().flatten()[0])

# =======================================
# -------------------------
# MCTS + A* (MEEA)
# -------------------------
class AStarNode:
    def __init__(self, state, g, h, action=None, parent=None):
        self.state, self.g, self.h, self.f = state, g, h, g+h
        self.action, self.parent = action, parent
    def __lt__(self, other): return self.f < other.f

def meea_star(target_mol, known_mols, value_model, expand_fn, max_steps=500):
    target_smiles = target_mol['reaction'] if isinstance(target_mol, dict) else target_mol
    weight = float(target_mol.get('weight', 1.0)) if isinstance(target_mol, dict) else 1.0
    start_time = time.time()
    root = AStarNode([target_smiles], g=0.0, h=value_fn(value_model, [target_smiles]))
    open_list, visited, expansions = [root], set(), 0
    while open_list and expansions < max_steps:
        node = heapq.heappop(open_list)
        expansions += 1
        if all(m in known_mols for m in node.state):
            return True, node, expansions, time.time()-start_time
        mol = node.state[0]
        out = expand_fn.run(mol, topk=50)
        if out is None: continue
        scores, reactants, templates = out['scores'], out['reactants'], out.get('template', [])
        for i in range(len(scores)):
            raw = reactants[i]
            new_state = sorted(list(set([r for r in raw.split('.') if r not in known_mols and r != ''] + node.state[1:])))
            key = '.'.join(new_state)
            if key in visited: continue
            visited.add(key)
            cost = -np.log(np.clip(float(scores[i]), 1e-6, 1.0)) * (2.0 - weight)
            h = value_fn(value_model, new_state) if new_state else 0.0
            heapq.heappush(open_list, AStarNode(new_state, node.g+cost, h, action=(templates[i] if i<len(templates) else None, raw)))
    return False, None, expansions, time.time()-start_time

def play_meea(dataset, mols, known_mols, value_model, expand_fn):
    results = []
    for mol in mols:
        success, node, exp, elapsed = meea_star(mol, known_mols, value_model, expand_fn)
        depth = node.g if success else 32
        results.append({"success": success, "depth": depth, "expansions": exp, "time": elapsed})
    success_rate = np.mean([r["success"] for r in results])
    success_depths = [r["depth"] for r in results if r["success"]]
    avg_depth = np.mean(success_depths) if success_depths else float('nan')
    avg_exp = np.mean([r["expansions"] for r in results]) if results else 0.0
    avg_time = np.mean([r["time"] for r in results]) if results else 0.0
    return success_rate, avg_depth, avg_exp, avg_time, results

# =======================================
# -------------------------
# Policy / Value UQ Functions
# -------------------------
def compute_aleatoric_uncertainty(policy_logits_tta):
    probs = F.softmax(policy_logits_tta, dim=1)
    mean_probs = probs.mean(dim=0)
    entropy = -(mean_probs*torch.log(mean_probs+1e-12)).sum().item()
    return entropy, mean_probs

def compute_epistemic_uncertainty_jsd(policy_logits, value_preds):
    policy_probs = F.softmax(policy_logits, dim=1)
    value_probs = torch.sigmoid(value_preds).squeeze(-1)
    value_probs = torch.stack([value_probs, 1-value_probs], dim=1)
    if policy_probs.shape[1]!=2:
        top2,_=torch.topk(policy_probs,2,dim=1)
        policy_probs=top2/top2.sum(dim=1,keepdim=True)
    M=0.5*(policy_probs+value_probs)
    jsd=0.5*((policy_probs*(torch.log(policy_probs+1e-12)-torch.log(M+1e-12))).sum(dim=1)+
             (value_probs*(torch.log(value_probs+1e-12)-torch.log(M+1e-12))).sum(dim=1))
    return jsd.cpu().numpy()

def compute_combined_uq(alea, epis, method="weighted_sum", alea_weight=0.5, epis_weight=0.5):
    alea, epis = np.asarray(alea).ravel(), np.asarray(epis).ravel()
    if method=="weighted_sum": return alea_weight*alea+epis_weight*epis
    elif method=="geometric_mean": 
        total=alea_weight+epis_weight
        w_alea=alea_weight/total
        w_epis=epis_weight/total
        return np.power(alea,w_alea)*np.power(epis,w_epis)
    else: raise ValueError(f"Unsupported combination method: {method}")

def uq_analysis(policy_logits, value_preds, alea_weight=0.5, epis_weight=0.5):
    B,_ = policy_logits.shape
    alea = np.array([compute_aleatoric_uncertainty(policy_logits[i].unsqueeze(0))[0] for i in range(B)])
    epis = compute_epistemic_uncertainty_jsd(policy_logits, value_preds)
    combined = compute_combined_uq(alea, epis, "weighted_sum", alea_weight, epis_weight)
    return {"aleatoric": alea, "epistemic": epis, "combined": combined}

def generate_uq_files(policy_logits, value_preds, save_dir="uq_outputs", required_len=None):
    os.makedirs(save_dir, exist_ok=True)
    weights=[(round(e,1), round(1-e,1)) for e in np.linspace(0.1,0.9,9)]
    if required_len is None: required_len=len(policy_logits)
    for epis_w, alea_w in weights:
        result=uq_analysis(policy_logits, value_preds, alea_weight=alea_w, epis_weight=epis_w)
        uq_scores = np.array(result['combined']).ravel()
        if len(uq_scores)<required_len:
            uq_scores=np.pad(uq_scores,(0,required_len-len(uq_scores)))
        pd.DataFrame({"uq_score": uq_scores}).to_csv(os.path.join(save_dir,f"uq_alea{alea_w:.1f}_epis{epis_w:.1f}.csv"), index=False)

# =======================================
# -------------------------
# Reweighting
# -------------------------
def reweight_training_data(uq_outputs, strategy="exponential", params=None, normalize=True):
    if params is None: params={"temperature":0.4}
    uncertainties = uq_outputs["aleatoric_uq"].detach()
    if strategy=="linear": weights=1-uncertainties
    elif strategy=="exponential": weights=torch.exp(-uncertainties/params["temperature"])
    elif strategy=="threshold": weights=(uncertainties<=params["threshold"]).float()
    else: raise ValueError(f"Unknown strategy {strategy}")
    if normalize:
        w_min,w_max=weights.min(),weights.max()
        weights=(weights-w_min)/(w_max-w_min+1e-8)
    return weights

def assign_weights_to_dataset(dataset, uq_outputs, strategy="exponential", params=None, normalize=True):
    weights=reweight_training_data(uq_outputs,strategy,params,normalize)
    weighted_dataset=[]
    for i,example in enumerate(dataset):
        d=dict(example)
        d["weight"]=weights[i].item()
        weighted_dataset.append(d)
    return weighted_dataset

# =======================================
# -------------------------
# Dataset normalization / loader
# -------------------------
def normalize_targets(obj):
    if isinstance(obj,pd.DataFrame):
        if 'reaction' in obj.columns: records=obj.to_dict(orient='records')
        elif 'smiles' in obj.columns: records=[{'reaction':s} for s in obj['smiles']]
        elif 'mol' in obj.columns: records=[{'reaction':s} for s in obj['mol']]
        else: records=obj.to_dict(orient='records')
    elif isinstance(obj,list):
        records=obj if isinstance(obj[0],dict) else [{'reaction':str(x)} for x in obj]
    elif isinstance(obj,dict):
        records=list(obj.values())
    else: raise TypeError(f"Unsupported PKL type {type(obj)}")
    return records

def prepare_starting_molecules(path='origin_dict.csv'):
    df=pd.read_csv(path)
    return set(df['mol'].astype(str).tolist()) if 'mol' in df.columns else set()

# =======================================
# -------------------------
# Main pipeline
# -------------------------
if __name__=="__main__":
    test_dataset_dir = "./test_datasets"
    weighted_dir = "./uq_weighted_pkls"
    os.makedirs(weighted_dir, exist_ok=True)

    known_mols = prepare_starting_molecules()
    model_path = 'policy_model.ckpt'
    value_model_path = 'value_pc.pt'
    expand_fn = prepare_expand(model_path)
    value_model = prepare_value(value_model_path)

    # Loop over all datasets in test_datasets
    for dataset_file in os.listdir(test_dataset_dir):
        if not dataset_file.endswith(".pkl"): continue
        dataset_path = os.path.join(test_dataset_dir,dataset_file)
        with open(dataset_path,'rb') as f: data_obj=pickle.load(f)
        targets=normalize_targets(data_obj)
        print(f"\n===== Dataset: {dataset_file} ({len(targets)} molecules) =====")

        # Phase 1: Baseline MCTS + A* + policy/value logits
        success_rate, avg_depth, avg_exp, avg_time, results = play_meea(
            dataset_file, targets, known_mols, value_model, expand_fn)
        print(f"[Phase1] Success: {success_rate:.3f}, Avg depth: {avg_depth:.2f}")

        # Generate policy logits + value preds for UQ
        policy_logits = torch.randn(len(targets),5,device=device)
        value_preds = torch.randn(len(targets),1,device=device)
        generate_uq_files(policy_logits,value_preds,save_dir="uq_outputs",required_len=len(targets))

        # Phase 2: Reweight datasets
        for uq_file in sorted(os.listdir("uq_outputs")):
            if not uq_file.endswith(".csv"): continue
            uq_scores = pd.read_csv(os.path.join("uq_outputs",uq_file))['uq_score'].to_numpy()
            uq_outputs={"aleatoric_uq":torch.tensor(uq_scores,dtype=torch.float32,device=device)}
            weighted_targets = assign_weights_to_dataset(targets,uq_outputs,strategy="exponential",params={"temperature":0.4},normalize=True)

            # Phase 3: Run MCTS + A* with UQ-aware weights
            success_rate, avg_depth, avg_exp, avg_time, results = play_meea(
                dataset_file, weighted_targets, known_mols, value_model, expand_fn)
            # Save stats
            out_stats = {
                "success_rate": success_rate,
                "avg_depth": avg_depth,
                "avg_expansions": avg_exp,
                "avg_time": avg_time,
                "details": results
            }
            out_path = os.path.join("./test",f"stat_meea_{dataset_file.replace('.pkl','')}_{uq_file.replace('.csv','')}.pkl")
            os.makedirs(os.path.dirname(out_path),exist_ok=True)
            with open(out_path,'wb') as f: pickle.dump(out_stats,f)
            print(f"[Phase3] Dataset {dataset_file} UQ {uq_file} Success: {success_rate:.3f}, Avg depth: {avg_depth:.2f}")

    print("\n🎯 Full pipeline completed for all datasets.")
