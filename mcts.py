#!/usr/bin/env python3
# meea_full_pipeline_gpu.py
# Full MCTS + A* + UQ aware pipeline for Lambda Labs
# Loops over all datasets in test_datasets folder
import os, sys, time, pickle, heapq, re
import numpy as np
import pandas as pd
import torch
import multiprocessing
from rdkit import Chem
from rdkit.Chem import AllChem
from valueEnsemble import ValueEnsemble
from policyNet import MLPModel

# ---------------------------
# SmilesEnumerator (TTA)
# ---------------------------
class SmilesEnumerator(object):
    def __init__(self, charset='@C)(=cOn1S2/H[N]\\', pad=120, leftpad=True, isomericSmiles=True, enum=True, canonical=False):
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
        self._char_to_int = dict((c,i) for i,c in enumerate(charset))
        self._int_to_char = dict((i,c) for i,c in enumerate(charset))

    def randomize_smiles(self, smiles):
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
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
        return np.zeros(fp_dim, dtype=np.bool_)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool_)
    arr[list(fp.GetOnBits())] = 1
    return arr

def batch_smiles_to_fp(s_list, fp_dim=2048):
    return np.array([smiles_to_fp(s, fp_dim) for s in s_list])

# ---------------------------
# GPU-aware model loaders
# ---------------------------
def prepare_expand(model_path, device):
    return MLPModel(model_path, '/content/template_rules.dat', device=device)

def prepare_value(model_f, device):
    model = ValueEnsemble(2048, 128, 0.1).to(device)
    ckpt = torch.load(model_f, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    return model

# ---------------------------
# Value function
# ---------------------------
def value_fn(model, mols, device):
    fps = batch_smiles_to_fp(mols, fp_dim=2048).astype(np.float32)
    num_mols = len(fps)
    if num_mols < 5:
        mask = np.ones(5, dtype=np.float32)
        fps_input = np.zeros((5, 2048), dtype=np.float32)
        fps_input[:num_mols,:] = fps
        mask[num_mols:] = 0
    else:
        mask = np.ones(num_mols, dtype=np.float32)
        fps_input = fps
    fps_tensor = torch.from_numpy(fps_input).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)
    with torch.no_grad():
        v = model(fps_tensor, mask_tensor).cpu().numpy()
    return float(v.flatten()[0])

# ---------------------------
# TTA policy logits
# ---------------------------
def _logmeanexp(a, axis=0):
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).mean(axis=axis, keepdims=True))).squeeze(axis)

def save_policy_outputs(expand_fn, mols, device, out_path, n_aug=20, topk=50, aggregation='mean', save_raw=True):
    sm_en = SmilesEnumerator(enum=True, canonical=False)
    results, raw_results = [], []
    with torch.no_grad():
        for idx, s in enumerate(mols):
            aug_scores_list, aug_candidates_list = [], []
            for a in range(n_aug):
                try: aug_s = sm_en.randomize_smiles(s)
                except: aug_s = s
                out = expand_fn.run(aug_s, topk=topk)
                if out is None:
                    aug_scores_list.append(np.array([]))
                    aug_candidates_list.append([])
                    continue
                aug_scores_list.append(np.array(out['scores'], dtype=np.float32))
                aug_candidates_list.append(list(out['reactants']))
            # union of candidates
            union_set = []
            for cands in aug_candidates_list:
                for c in cands:
                    if c not in union_set: union_set.append(c)
            if len(union_set) == 0: continue
            aligned = np.zeros((len(aug_scores_list), len(union_set)), dtype=np.float32)
            for i,(cands,scores) in enumerate(zip(aug_candidates_list, aug_scores_list)):
                for j,cand in enumerate(cands):
                    k = union_set.index(cand)
                    aligned[i,k] = scores[j]
            if aggregation=='mean': agg_scores = aligned.mean(axis=0)
            elif aggregation=='logmeanexp': agg_scores = np.exp(_logmeanexp(np.log(np.clip(aligned,1e-12,1.0)), axis=0))
            results.append({'mol':s, 'candidates':union_set, 'agg_scores':agg_scores})
            if save_raw: raw_results.append({'mol':s, 'union_candidates':union_set, 'aligned_scores':aligned})
            if (idx+1) % 100 == 0: print(f"[INFO] processed {idx+1}/{len(mols)} molecules")
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.save(out_path, np.array(results, dtype=object))
    if save_raw:
        raw_path = out_path.replace(".npy","_raw.npy")
        np.save(raw_path, np.array(raw_results, dtype=object))
    return results

def save_value_outputs(value_model, mols, device, out_path):
    preds = [value_fn(value_model,[s], device) for s in mols]
    df = pd.DataFrame({'mol':mols,'value':preds})
    df.to_csv(out_path,index=False)
    return preds

# ---------------------------
# UQ computation
# ---------------------------
def compute_aleatoric(policy_logits):
    probs = torch.softmax(torch.tensor(policy_logits).unsqueeze(0), dim=1)
    mean_probs = probs.mean(dim=0)
    entropy = -(mean_probs * torch.log(mean_probs + 1e-12)).sum().item()
    return entropy

def compute_epistemic(policy_logits, value_preds):
    policy_probs = torch.softmax(torch.tensor([p['agg_scores'] for p in policy_logits]), dim=1)
    value_probs = torch.sigmoid(torch.tensor(value_preds).unsqueeze(-1))
    value_probs = torch.cat([value_probs, 1-value_probs], dim=1)
    if policy_probs.shape[1]!=2:
        top2,_ = torch.topk(policy_probs,2,dim=1)
        policy_probs = top2 / top2.sum(dim=1,keepdim=True)
    M = 0.5*(policy_probs+value_probs)
    jsd = 0.5*((policy_probs*(torch.log(policy_probs+1e-12)-torch.log(M+1e-12))).sum(dim=1)
              + (value_probs*(torch.log(value_probs+1e-12)-torch.log(M+1e-12))).sum(dim=1))
    return jsd.cpu().numpy()

def compute_combined_uq(alea, epis, alea_w=0.5, epis_w=0.5):
    return (alea_w*np.array(alea) + epis_w*np.array(epis)).ravel()

def reweight_training_data_exponential(uq_scores):
    # higher UQ -> higher weight
    uq_scores = np.array(uq_scores)
    weights = np.exp(uq_scores / np.max(uq_scores))
    weights = weights / np.mean(weights)
    return weights

# ---------------------------
# MCTS + A* Search
# ---------------------------
class AStarNode:
    def __init__(self,state,g,h,action=None,parent=None):
        self.state,self.g,self.h,self.action,self.parent = state,g,h,action,parent
        self.f = g+h
    def __lt__(self,other): return self.f<other.f

def meea_star(target_mol, known_mols, value_model, expand_fn, device, max_steps=500):
    start_time = time.time()
    root_h = value_fn(value_model,[target_mol], device)
    root = AStarNode([target_mol],0,root_h)
    open_list, visited, expansions = [], set(), 0
    heapq.heappush(open_list, root)
    while open_list and expansions<max_steps:
        node = heapq.heappop(open_list)
        expansions+=1
        if all(m in known_mols for m in node.state):
            return True,node,expansions,time.time()-start_time
        expanded_mol = node.state[0]
        expanded_policy = expand_fn.run(expanded_mol, topk=50)
        if expanded_policy is None: continue
        for i in range(len(expanded_policy['scores'])):
            reactant = [r for r in expanded_policy['reactants'][i].split('.') if r not in known_mols]
            reactant = sorted(list(set(reactant + node.state[1:])))
            state_key = '.'.join(reactant)
            if state_key in visited: continue
            visited.add(state_key)
            cost = -np.log(np.clip(expanded_policy['scores'][i],1e-6,1.0))
            h = value_fn(value_model, reactant, device) if reactant else 0
            child = AStarNode(reactant,node.g+cost,h,parent=node,action=(expanded_policy['template'][i],expanded_policy['reactants'][i]))
            heapq.heappush(open_list,child)
    return False,None,expansions,time.time()-start_time

def play_meea(dataset,mols,known_mols,value_model,expand_fn,device):
    results=[]
    total=len(mols)
    for i,mol in enumerate(mols):
        success,node,expansions,elapsed = meea_star(mol,known_mols,value_model,expand_fn,device)
        depth = node.g if success else 32
        results.append({'success':success,'depth':depth,'expansions':expansions,'time':elapsed})
        pct=(i+1)/total*100
        sys.stdout.write(f"\rDataset {dataset}: {i+1}/{total} molecules ({pct:.1f}%) done")
        sys.stdout.flush()
    print()
    success_rate = np.mean([r['success'] for r in results])
    avg_depth = np.mean([r['depth'] for r in results if r['success']]) if any(r['success'] for r in results) else 0
    avg_exp = np.mean([r['expansions'] for r in results])
    avg_time = np.mean([r['time'] for r in results])
    return success_rate, avg_depth, avg_exp, avg_time, results

# ---------------------------
# Dataset preparation
# ---------------------------
def normalize_targets(obj):
    if isinstance(obj,list):
        if len(obj)==0: return []
        if isinstance(obj[0],dict): return obj
        return [{'reaction':str(x)} for x in obj]
    elif isinstance(obj,pd.DataFrame):
        if 'reaction' in obj.columns: return obj.to_dict(orient='records')
        elif 'mol' in obj.columns: return [{'reaction':str(x)} for x in obj['mol']]
    elif isinstance(obj,dict): return list(obj.values())
    else: raise TypeError(f"Unsupported dataset type: {type(obj)}")

def prepare_starting_molecules(path='origin_dict.csv'):
    df = pd.read_csv(path)
    return set(df['mol'].astype(str).tolist()) if 'mol' in df.columns else set()

# ---------------------------
# Main loop
# ---------------------------
if __name__=='__main__':
    multiprocessing.set_start_method('spawn', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    known_mols = prepare_starting_molecules()
    model_path = '/content/policy_model.ckpt'
    model_f = '/content/value_pc.pt'

    expand_fn = prepare_expand(model_path, device)
    value_model = prepare_value(model_f, device)

    dataset_folder = './test_dataset'
    files = sorted([f for f in os.listdir(dataset_folder) if f.endswith('.pkl')])

    os.makedirs('./test', exist_ok=True)

    for f in files:
        path = os.path.join(dataset_folder,f)
        with open(path,'rb') as infile: raw_obj=pickle.load(infile)
        targets = normalize_targets(raw_obj)
        base_tag = os.path.splitext(f)[0]

        print(f"\n===== Running Phase 1: MCTS + A* on {base_tag} =====")
        # Phase 1: MCTS + A*, collect policy logits + value preds
        success_rate, avg_depth, avg_exp, avg_time, results = play_meea(base_tag,targets,known_mols,value_model,expand_fn,device)
        out_stats = {"success_rate":success_rate,"avg_depth":avg_depth,"avg_expansions":avg_exp,"avg_time":avg_time,"details":results}
        with open(f'./test/stat_meea_{base_tag}.pkl','wb') as outf: pickle.dump(out_stats,outf)

        # Save policy logits and value predictions
        policy_logits = save_policy_outputs(expand_fn, targets, device, out_path=f'./test/policy_logits_tta_{base_tag}.npy')
        value_preds = save_value_outputs(value_model, targets, device, out_path=f'./test/value_preds_{base_tag}.csv')

        # Phase 2: compute UQ
        alea = [compute_aleatoric(p['agg_scores']) for p in policy_logits]
        epis = compute_epistemic(policy_logits, value_preds)
        uq_scores = compute_combined_uq(alea,epis)
        weights = reweight_training_data_exponential(uq_scores)

        # Phase 3: reweight targets and rerun MCTS + A*
        weighted_targets=[]
        for t,w in zip(targets,weights):
            t_copy = t.copy() if isinstance(t,dict) else {'reaction':str(t)}
            t_copy['weight']=float(w)
            weighted_targets.append(t_copy)

        print(f"\n===== Running Phase 3: Weighted MCTS + A* on {base_tag} =====")
        success_rate, avg_depth, avg_exp, avg_time, results = play_meea(base_tag,weighted_targets,known_mols,value_model,expand_fn,device)
        out_stats = {"success_rate":success_rate,"avg_depth":avg_depth,"avg_expansions":avg_exp,"avg_time":avg_time,"details":results}
        with open(f'./test/stat_meea_weighted_{base_tag}.pkl','wb') as outf: pickle.dump(out_stats,outf)

        print(f"📊 Completed {base_tag}: success_rate={success_rate:.3f}, avg_depth={avg_depth:.2f}, avg_exp={avg_exp:.2f}, avg_time={avg_time:.2f}s\n")
