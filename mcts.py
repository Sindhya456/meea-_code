#!/usr/bin/env python3
# ====================================================
# Full MCTS + A* + UQ-aware pipeline (GPU-ready)
# Loops over all datasets in test_dataset
# Saves results as TXT for shell access
# ====================================================
#!/usr/bin/env python3#!/usr/bin/env python3
# ====================================================
# Full MCTS + A* + UQ-aware pipeline (GPU-ready)
# Loops over all datasets in test_dataset
# ====================================================

import os, sys, time, pickle
import torch, numpy as np, pandas as pd
import heapq
from tqdm import tqdm

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
# Fingerprints & value function
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
    # fix path to saved_model folder
    return MLPModel(model_path, './saved_model/template_rules.dat', device=device)

def prepare_value(model_f, device='cuda'):
    model = ValueEnsemble(2048, 128, 0.1).to(device)
    model.load_state_dict(torch.load(model_f, map_location=device))
    model.eval()
    return model

# ---------------------------
# Phase 1: Baseline MCTS + A*
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

    while open_list and expansions<max_steps:
        node = heapq.heappop(open_list)
        expansions +=1
        if all(m in known_mols for m in node.state):
            return True, node, expansions, time.time()-start
        # Expand
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
# Phase 2: UQ calculation
# ---------------------------
def compute_aleatoric(policy_out):
    logits = torch.tensor(policy_out['scores']).unsqueeze(0)
    probs = torch.softmax(logits, dim=1)
    mean_probs = probs.mean(dim=0)
    return -(mean_probs*torch.log(mean_probs+1e-12)).sum().item()

def compute_epistemic(policy_outs, value_preds):
    all_probs=[]
    for po in policy_outs:
        logits = torch.tensor(po['scores']).unsqueeze(0)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs)
    all_probs=torch.cat(all_probs,dim=0)
    value_probs = torch.sigmoid(torch.tensor(value_preds)).squeeze(-1)
    value_probs = torch.stack([value_probs,1-value_probs],dim=1)
    if all_probs.shape[1]!=2:
        top2,_=torch.topk(all_probs,2,dim=1)
        all_probs=top2/top2.sum(1,keepdim=True)
    M=0.5*(all_probs+value_probs)
    jsd=0.5*((all_probs*(torch.log(all_probs+1e-12)-torch.log(M+1e-12))).sum(1)+
             (value_probs*(torch.log(value_probs+1e-12)-torch.log(M+1e-12))).sum(1))
    return jsd.numpy()

def generate_uq_files(mols, expand_fn, value_model, device, save_dir="uq_outputs"):
    os.makedirs(save_dir,exist_ok=True)
    policy_outs=[]
    value_preds=[]
    for mol in tqdm(mols, desc="Phase2 UQ"):
        po = expand_fn.run(mol, topk=50)
        policy_outs.append(po)
        v = value_fn(value_model, [mol], device)
        value_preds.append(v[0])
    # compute combined UQ for 9 weight pairs
    weights=[(round(e,1),round(1-e,1)) for e in np.linspace(0.1,0.9,9)]
    for alea_w, epis_w in weights:
        alea=[compute_aleatoric(po) for po in policy_outs]
        epis=compute_epistemic(policy_outs,value_preds)
        combined = np.array(alea_w)*np.array(alea)+np.array(epis_w)*np.array(epis)
        fname=f"uq_alea{alea_w:.1f}_epis{epis_w:.1f}.txt"
        pd.DataFrame({"uq_score":combined}).to_csv(os.path.join(save_dir,fname), index=False, sep='\t')

# ---------------------------
# Phase 2b: Weighted dataset creation
# ---------------------------
def create_weighted_dataset(mols, uq_scores, method='exponential', scale=5.0):
    uq_scores=np.array(uq_scores).ravel()
    if method=='linear':
        weights = uq_scores/uq_scores.sum()
    elif method=='exponential':
        exp_scores = np.exp(scale*uq_scores)
        weights = exp_scores/exp_scores.sum()
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    indices = np.random.choice(len(mols), size=len(mols), p=weights)
    weighted_mols = [mols[i] for i in indices]
    return weighted_mols, weights

# ---------------------------
# Save stats to human-readable txt
# ---------------------------
def save_stats_txt(filename, dataset_name, results):
    success_rate = np.mean([r["success"] for r in results])
    avg_depth = np.mean([r["depth"] for r in results if r["success"]]) if any(r["success"] for r in results) else 0
    avg_exp = np.mean([r["expansions"] for r in results])
    avg_time = np.mean([r["time"] for r in results])
    with open(filename,'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Success rate: {success_rate:.3f}\n")
        f.write(f"Avg depth: {avg_depth:.2f}\n")
        f.write(f"Avg expansions: {avg_exp:.2f}\n")
        f.write(f"Avg time: {avg_time:.2f}s\n")

# ---------------------------
# Main pipeline
# ---------------------------
if __name__=='__main__':
    os.makedirs('./test',exist_ok=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'

    expand_fn = prepare_expand('./saved_model/policy_model.ckpt', device=device)
    value_model = prepare_value('./saved_model/value_pc.pt', device=device)

    # clear old outputs in test folder
    for f in os.listdir('./test'):
        os.remove(os.path.join('./test',f))

    # Loop over datasets
    datasets=os.listdir('./test_dataset')
    for ds_file in datasets:
        ds_path=os.path.join('./test_dataset',ds_file)
        with open(ds_path,'rb') as f:
            mols=pickle.load(f)
        ds_name=os.path.splitext(ds_file)[0]
        print(f"[INFO] Dataset: {ds_name}, molecules: {len(mols)}")

        # Phase 1
        baseline_results=[]
        for mol in tqdm(mols, desc=f"Phase1 MCTS+A* ({ds_name})"):
            success,node,exp_count,elapsed=meea_star(mol,mols,value_model,expand_fn,device)
            depth=node.g if success else 32
            baseline_results.append({"success":success,"depth":depth,"expansions":exp_count,"time":elapsed})
        save_stats_txt(f'./test/stat_baseline_{ds_name}.txt', ds_name, baseline_results)

        # Phase 2: UQ files
        generate_uq_files(mols, expand_fn, value_model, device, save_dir='./uq_outputs')

        # Phase 2b: Weighted dataset
        uq_file='./uq_outputs/uq_alea0.5_epis0.5.txt'
        uq_scores=pd.read_csv(uq_file, sep='\t')['uq_score'].to_numpy()
        weighted_mols, weights = create_weighted_dataset(mols, uq_scores)

        # Phase 3: Weighted MCTS+A*
        weighted_results=[]
        for mol in tqdm(weighted_mols, desc=f"Phase3 Weighted MCTS+A* ({ds_name})"):
            success,node,exp_count,elapsed=meea_star(mol,mols,value_model,expand_fn,device)
            depth=node.g if success else 32
            weighted_results.append({"success":success,"depth":depth,"expansions":exp_count,"time":elapsed})
        save_stats_txt(f'./test/stat_meea_{ds_name}.txt', ds_name, weighted_results)

    print("[INFO] Pipeline complete for all datasets!")

# ====================================================
# Full MCTS + A* + UQ-aware pipeline (GPU-ready)
# Loops over all datasets in test_dataset
# Results saved as TXT for easy shell access
# ====================================================

import os, sys, time, pickle
import torch, numpy as np, pandas as pd
import heapq
from tqdm import tqdm

# ---------------------------
# MCTS + A* helpers
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
    # FIXED: path to template_rules.dat
    template_rules_path = './saved_model/template_rules.dat'
    return MLPModel(model_path, template_rules_path, device=device)

def prepare_value(model_f, device='cuda'):
    model = ValueEnsemble(2048, 128, 0.1).to(device)
    model.load_state_dict(torch.load(model_f, map_location=device))
    model.eval()
    return model

# ---------------------------
# Phase 1: Baseline MCTS + A* run
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

    while open_list and expansions<max_steps:
        node = heapq.heappop(open_list)
        expansions +=1
        if all(m in known_mols for m in node.state):
            return True, node, expansions, time.time()-start
        # Expand
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
# Phase 2: UQ calculation (policy logits + value preds)
# ---------------------------
def compute_aleatoric(policy_logits):
    probs = torch.softmax(policy_logits, dim=1)
    mean_probs = probs.mean(dim=0)
    return -(mean_probs*torch.log(mean_probs+1e-12)).sum().item()

def compute_epistemic(policy_logits, value_preds):
    policy_probs = torch.softmax(policy_logits, dim=1)
    value_probs = torch.sigmoid(value_preds).squeeze(-1)
    value_probs = torch.stack([value_probs,1-value_probs],dim=1)
    if policy_probs.shape[1]!=2:
        top2,_=torch.topk(policy_probs,2,dim=1)
        policy_probs=top2/top2.sum(1,keepdim=True)
    M=0.5*(policy_probs+value_probs)
    jsd=0.5*((policy_probs*(torch.log(policy_probs+1e-12)-torch.log(M+1e-12))).sum(1)+
             (value_probs*(torch.log(value_probs+1e-12)-torch.log(M+1e-12))).sum(1))
    return jsd.cpu().numpy()

def compute_combined_uq(alea, epis, w_alea=0.5, w_epis=0.5):
    return w_alea*np.array(alea).ravel()+w_epis*np.array(epis).ravel()

def generate_uq_files(policy_logits, value_preds, save_dir="uq_outputs", required_len=None):
    os.makedirs(save_dir, exist_ok=True)
    weights = [(round(e,1), round(1-e,1)) for e in np.linspace(0.1,0.9,9)]
    if required_len is None: required_len = len(policy_logits)
    if len(policy_logits)!=required_len:
        raise ValueError(f"Expected {required_len} samples, got {len(policy_logits)}")
    for epis_w, alea_w in weights:
        alea=[]
        for i in range(len(policy_logits)):
            ent=compute_aleatoric(policy_logits[i].unsqueeze(0))
            alea.append(ent)
        epis=compute_epistemic(policy_logits, value_preds)
        combined=compute_combined_uq(alea, epis, w_alea=alea_w, w_epis=epis_w)
        filename=f"uq_alea{alea_w:.1f}_epis{epis_w:.1f}.csv"
        pd.DataFrame({"uq_score":combined}).to_csv(os.path.join(save_dir,filename), index=False)

# ---------------------------
# Phase 2b: Weighted dataset creation
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
    os.makedirs('./test',exist_ok=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'

    # Load models
    expand_fn = prepare_expand('./saved_model/policy_model.ckpt', device=device)
    value_model = prepare_value('./saved_model/value_pc.pt', device=device)

    # Loop over all datasets in test_dataset
    datasets=os.listdir('./test_dataset')
    for ds_file in datasets:
        ds_path=os.path.join('./test_dataset',ds_file)
        with open(ds_path,'rb') as f: mols=pickle.load(f)
        ds_name=os.path.splitext(ds_file)[0]

        print(f"[INFO] Dataset: {ds_name}, molecules: {len(mols)}")

        # ---------- Phase 1: Baseline MCTS + A* ----------
        baseline_results=[]
        for mol in tqdm(mols, desc=f"Phase1 MCTS+A* ({ds_name})"):
            success,node,exp_count,elapsed = meea_star(mol, mols, value_model, expand_fn, device)
            depth=node.g if success else 32
            baseline_results.append({"success":success,"depth":depth,"expansions":exp_count,"time":elapsed})
        # Save baseline TXT
        out_path=f'./test/stat_baseline_{ds_name}.txt'
        with open(out_path,'w') as f:
            success_rate = np.mean([r["success"] for r in baseline_results])
            avg_depth = np.mean([r["depth"] for r in baseline_results if r["success"]]) if any(r["success"] for r in baseline_results) else 0
            avg_exp = np.mean([r["expansions"] for r in baseline_results])
            avg_time = np.mean([r["time"] for r in baseline_results])
            f.write(f"Dataset: {ds_name}\n")
            f.write(f"Success rate: {success_rate:.3f}\n")
            f.write(f"Avg depth: {avg_depth:.2f}\n")
            f.write(f"Avg expansions: {avg_exp:.2f}\n")
            f.write(f"Avg time: {avg_time:.2f}s\n")
        print(f"[INFO] Phase 1 results saved: {out_path}")

        # ---------- Phase 2: Generate UQ files (real logits + value preds) ----------
        # Build real policy logits for each molecule
        policy_logits_list=[]
        for m in tqdm(mols, desc=f"Generating policy logits ({ds_name})"):
            out = expand_fn.run(m)
            if out is None or len(out['scores'])==0:
                scores = torch.zeros(5)
            else:
                scores = torch.tensor(out['scores'][:5])
                if len(scores)<5: scores = torch.cat([scores, torch.zeros(5-len(scores))])
            policy_logits_list.append(scores)
        policy_logits = torch.stack(policy_logits_list).to(device)

        value_preds = torch.tensor([value_fn(value_model,[m],device)[0] for m in mols]).to(device)

        generate_uq_files(policy_logits,value_preds, save_dir='./uq_outputs', required_len=len(mols))

        # ---------- Phase 2b: Create weighted dataset ----------
        uq_csv='./uq_outputs/uq_alea0.5_epis0.5.csv'
        uq_scores=pd.read_csv(uq_csv)['uq_score'].to_numpy()
        weighted_mols, weights = create_weighted_dataset(mols, uq_scores)

        # ---------- Phase 3: MCTS + A* on weighted dataset ----------
        weighted_results=[]
        for mol in tqdm(weighted_mols, desc=f"Phase3 Weighted MCTS+A* ({ds_name})"):
            success,node,exp_count,elapsed = meea_star(mol, mols, value_model, expand_fn, device)
            depth=node.g if success else 32
            weighted_results.append({"success":success,"depth":depth,"expansions":exp_count,"time":elapsed})

        # Save Phase 3 TXT
        out_path=f'./test/stat_meea_{ds_name}.txt'
        with open(out_path,'w') as f:
            success_rate = np.mean([r["success"] for r in weighted_results])
            avg_depth = np.mean([r["depth"] for r in weighted_results if r["success"]]) if any(r["success"] for r in weighted_results) else 0
            avg_exp = np.mean([r["expansions"] for r in weighted_results])
            avg_time = np.mean([r["time"] for r in weighted_results])
            f.write(f"Dataset: {ds_name}\n")
            f.write(f"Success rate: {success_rate:.3f}\n")
            f.write(f"Avg depth: {avg_depth:.2f}\n")
            f.write(f"Avg expansions: {avg_exp:.2f}\n")
            f.write(f"Avg time: {avg_time:.2f}s\n")
        print(f"[INFO] Phase 3 results saved: {out_path}")

    print("[INFO] Pipeline complete for all datasets!")

import os, sys, time, pickle
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

def generate_uq_files(policy_logits, value_preds, save_dir="uq_outputs", required_len=190):
    os.makedirs(save_dir, exist_ok=True)
    weights = [(round(e,1), round(1-e,1)) for e in np.linspace(0.1,0.9,9)]
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load models
    expand_fn = prepare_expand('./saved_model/policy_model.ckpt', device=device)
    value_model = prepare_value('./saved_model/value_pc.pt', device=device)

    # Loop over datasets
    datasets = os.listdir('./test_dataset')
    for ds_file in datasets:
        ds_path = os.path.join('./test_dataset', ds_file)
        with open(ds_path,'rb') as f: mols = pickle.load(f)
        ds_name = os.path.splitext(ds_file)[0]
        known_mols = set(mols)

        print(f"[INFO] Dataset: {ds_name}, molecules: {len(mols)}")

        # Phase 1: Baseline MCTS + A*
        baseline_results=[]
        for mol in tqdm(mols, desc=f"Phase1 MCTS+A* ({ds_name})"):
            success,node,exp_count,elapsed = meea_star(mol, known_mols, value_model, expand_fn, device)
            depth = node.g if success else 32
            baseline_results.append({"success":success,"depth":depth,"expansions":exp_count,"time":elapsed})

        # Save baseline as TXT
        out_path = f'./test/stat_baseline_{ds_name}.txt'
        with open(out_path,'w') as f:
            for r in baseline_results:
                f.write(f"{r['success']}\t{r['depth']:.2f}\t{r['expansions']}\t{r['time']:.4f}\n")
        print(f"[INFO] Phase 1 results saved: {out_path}")

        # Phase 2: Generate UQ files (use real model logits)
        policy_logits = torch.stack([expand_fn.run(m) for m in mols]).to(device)  # replace placeholder
        value_preds = torch.tensor([value_fn(value_model,[m],device)[0] for m in mols]).to(device)
        generate_uq_files(policy_logits, value_preds, save_dir='./uq_outputs', required_len=len(mols))

        # Phase 2b: Weighted dataset
        uq_file = './uq_outputs/uq_alea0.5_epis0.5.txt'
        uq_scores = np.loadtxt(uq_file)
        weighted_mols, weights = create_weighted_dataset(mols, uq_scores)

        # Phase 3: MCTS + A* on weighted dataset
        weighted_results=[]
        for mol in tqdm(weighted_mols, desc=f"Phase3 Weighted MCTS+A* ({ds_name})"):
            success,node,exp_count,elapsed = meea_star(mol, known_mols, value_model, expand_fn, device)
            depth = node.g if success else 32
            weighted_results.append({"success":success,"depth":depth,"expansions":exp_count,"time":elapsed})

        # Save Phase 3 as TXT
        out_path = f'./test/stat_meea_{ds_name}.txt'
        with open(out_path,'w') as f:
            for r in weighted_results:
                f.write(f"{r['success']}\t{r['depth']:.2f}\t{r['expansions']}\t{r['time']:.4f}\n")
        print(f"[INFO] Phase 3 results saved: {out_path}")

    print("[INFO] Pipeline complete for all datasets!")
