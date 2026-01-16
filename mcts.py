#!/usr/bin/env python3
# ====================================================
# Full GPU-aware MCTS + A* + UQ pipeline
# Loops over all datasets in ./test_dataset
# ====================================================

import os, pickle, time
import torch, numpy as np, pandas as pd
from tqdm import tqdm
from policyNet import MLPModel
from valueEnsemble import ValueEnsemble
from rdkit import Chem
from rdkit.Chem import AllChem
import heapq
import shutil
import signal
from contextlib import contextmanager

# --------------------------- Helper functions ---------------------------

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def prepare_starting_molecules():
    path = './prepare_data/origin_dict.csv'
    df = pd.read_csv(path)
    return set(df['mol'].tolist())

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

def prepare_expand(model_path, device='cuda'):
    template_path = './saved_model/template_rules.dat'
    return MLPModel(model_path, template_path, device=device)

def prepare_value(model_path, device='cuda'):
    model = ValueEnsemble(2048,128,0.1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# --------------------------- Phase 1 & 3: A* Search ---------------------------

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
    try:
        h0 = value_fn(value_model, [target], device)[0]
    except:
        h0 = 1.0
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
        if policy_out is None or len(policy_out['scores'])==0:
            continue
        for i in range(len(policy_out['scores'])):
            reactants = [r for r in policy_out['reactants'][i].split('.') if r not in known_mols]
            reactants = sorted(list(set(reactants + node.state[1:])))
            key = '.'.join(reactants)
            if key in visited: continue
            visited.add(key)
            cost = -np.log(np.clip(policy_out['scores'][i], 1e-3, 1.0))
            try:
                h = value_fn(value_model, reactants, device)[0] if reactants else 0
            except:
                h = 1.0
            child = AStarNode(reactants, g=node.g+cost, h=h, parent=node, action=(policy_out['template'][i], policy_out['reactants'][i]))
            heapq.heappush(open_list, child)
    return False, None, expansions, time.time()-start

# --------------------------- Phase 2: UQ ---------------------------

def compute_aleatoric(policy_logits):
    probs = torch.softmax(policy_logits, dim=1)
    mean_probs = probs.mean(dim=0)
    return -(mean_probs*torch.log(mean_probs+1e-12)).sum().item()

def compute_epistemic(policy_logits, value_preds):
    policy_probs = torch.softmax(policy_logits, dim=1)
    value_probs = torch.sigmoid(value_preds).squeeze(-1)
    value_probs = torch.stack([value_probs, 1-value_probs], dim=1)
    if policy_probs.shape[1] != 2:
        top2,_=torch.topk(policy_probs,2,dim=1)
        policy_probs=top2/top2.sum(1,keepdim=True)
    M=0.5*(policy_probs+value_probs)
    jsd=0.5*((policy_probs*(torch.log(policy_probs+1e-12)-torch.log(M+1e-12))).sum(1)+
             (value_probs*(torch.log(value_probs+1e-12)-torch.log(M+1e-12))).sum(1))
    return jsd.cpu().numpy()

def generate_uq_files(mols, expand_fn, value_model, device, save_dir="uq_outputs"):
    os.makedirs(save_dir, exist_ok=True)
    policy_logits, value_preds = [], []
    for mol in tqdm(mols, desc="Phase2 UQ"):
        policy_out = expand_fn.run(mol, topk=50)
        if policy_out is None or len(policy_out['scores'])==0:
            logits = torch.zeros(1,2)
        else:
            logits = torch.tensor(policy_out['scores']).unsqueeze(0)
        policy_logits.append(logits)
        try:
            v = value_fn(value_model, [mol], device)
            value_preds.append(torch.tensor([v]))
        except:
            value_preds.append(torch.tensor([0.0]))
    policy_logits = torch.cat(policy_logits, dim=0)
    value_preds = torch.cat(value_preds, dim=0)
    # save combined UQ score
    combined = compute_epistemic(policy_logits, value_preds)
    uq_csv = os.path.join(save_dir,"uq_combined.csv")
    pd.DataFrame({"uq_score":combined}).to_csv(uq_csv, index=False)
    return uq_csv

def create_weighted_dataset(mols, uq_scores):
    uq_scores = np.array(uq_scores).ravel()
    weights = np.exp(-uq_scores)
    weights /= weights.sum()
    indices = np.random.choice(len(mols), size=len(mols), p=weights)
    return [mols[i] for i in indices]

# --------------------------- Main Pipeline ---------------------------

if __name__=='__main__':
    # clear old test folder
    if os.path.exists('./test'):
        shutil.rmtree('./test')
    os.makedirs('./test', exist_ok=True)
    os.makedirs('./uq_outputs', exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    known_mols = prepare_starting_molecules()

    expand_fn = prepare_expand('./saved_model/policy_model.ckpt', device=device)
    value_model = prepare_value('./saved_model/value_pc.pt', device=device)

    datasets = os.listdir('./test_dataset')
    for ds_file in datasets:
        ds_path = os.path.join('./test_dataset', ds_file)
        with open(ds_path,'rb') as f:
            mols = pickle.load(f)
        ds_name = os.path.splitext(ds_file)[0]
        print(f"[INFO] Dataset: {ds_name}, molecules: {len(mols)}")

        # ---------- Phase 1: Baseline ----------
        baseline_results = []
        for mol in tqdm(mols, desc=f"Phase1 MCTS+A* ({ds_name})"):
            try:
                with time_limit(120):
                    success,node,exp_count,elapsed = meea_star(mol, known_mols, value_model, expand_fn, device)
            except:
                success,node,exp_count,elapsed = False,None,0,0
            depth = node.g if node else 32
            baseline_results.append({"success":success,"depth":depth,"expansions":exp_count,"time":elapsed})
        # compute stats
        success_rate = np.mean([r['success'] for r in baseline_results])
        avg_depth = np.mean([r['depth'] for r in baseline_results])
        avg_exp = np.mean([r['expansions'] for r in baseline_results])
        avg_time = np.mean([r['time'] for r in baseline_results])
        # save to txt
        out_txt = f'./test/stat_baseline_{ds_name}.txt'
        with open(out_txt,'w') as f:
            f.write(f"Dataset: {ds_name}\nSuccess rate: {success_rate:.3f}\nAvg depth: {avg_depth:.2f}\nAvg expansions: {avg_exp:.2f}\nAvg time: {avg_time:.2f}s\n")
        print(f"[INFO] Phase 1 results saved: {out_txt}")

        # ---------- Phase 2: UQ ----------
        uq_csv = generate_uq_files(mols, expand_fn, value_model, device, save_dir='./uq_outputs')
        uq_scores = pd.read_csv(uq_csv)['uq_score'].to_numpy()

        # ---------- Phase 2b: weighted dataset ----------
        weighted_mols = create_weighted_dataset(mols, uq_scores)

        # ---------- Phase 3: weighted MCTS + A* ----------
        weighted_results = []
        for mol in tqdm(weighted_mols, desc=f"Phase3 Weighted MCTS+A* ({ds_name})"):
            try:
                with time_limit(120):
                    success,node,exp_count,elapsed = meea_star(mol, known_mols, value_model, expand_fn, device)
            except:
                success,node,exp_count,elapsed = False,None,0,0
            depth = node.g if node else 32
            weighted_results.append({"success":success,"depth":depth,"expansions":exp_count,"time":elapsed})
        # compute stats
        success_rate = np.mean([r['success'] for r in weighted_results])
        avg_depth = np.mean([r['depth'] for r in weighted_results])
        avg_exp = np.mean([r['expansions'] for r in weighted_results])
        avg_time = np.mean([r['time'] for r in weighted_results])
        # save to txt
        out_txt = f'./test/stat_meea_{ds_name}.txt'
        with open(out_txt,'w') as f:
            f.write(f"Dataset: {ds_name}\nSuccess rate: {success_rate:.3f}\nAvg depth: {avg_depth:.2f}\nAvg expansions: {avg_exp:.2f}\nAvg time: {avg_time:.2f}s\n")
        print(f"[INFO] Phase 3 results saved: {out_txt}")

    print("[INFO] Pipeline complete for all datasets!")
