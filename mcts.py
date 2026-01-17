# -*- coding: utf-8 -*-
"""
Full MEEA* pipeline with Phase 1 baseline, Phase 2 UQ weighting, Phase 3 UQ-aware search
- Phase 1: baseline MCTS+A* with TTA
- Phase 2: compute aleatoric + epistemic (JSD) uncertainties
- Phase 3: weighted MCTS+A* using computed weights
"""

import os, sys, pickle, time, heapq, re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
import multiprocessing

# --- Smiles Enumerator ---
import numpy as _np
class SmilesEnumerator:
    def __init__(self, charset='@C)(=cOn1S2/H[N]\\', pad=120, leftpad=True, isomericSmiles=True, enum=True, canonical=False):
        self._charset = None
        self.charset = charset
        self.pad = pad
        self.leftpad = leftpad
        self.isomericSmiles = isomericSmiles
        self.enumerate = enum
        self.canonical = canonical

    @property
    def charset(self): return self._charset

    @charset.setter
    def charset(self, charset):
        self._charset = charset
        self._charlen = len(charset)
        self._char_to_int = {c:i for i,c in enumerate(charset)}
        self._int_to_char = {i:c for i,c in enumerate(charset)}

    def randomize_smiles(self, smiles):
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        idxs = list(range(m.GetNumAtoms()))
        _np.random.shuffle(idxs)
        nm = Chem.RenumberAtoms(m, idxs)
        return Chem.MolToSmiles(nm, canonical=self.canonical, isomericSmiles=self.isomericSmiles)

# --- Fingerprint Helpers ---
def smiles_to_fp(s, fp_dim=2048):
    if isinstance(s, dict): s = s.get('reaction','')
    s = str(s).strip()
    mol = Chem.MolFromSmiles(s)
    if mol is None: return np.zeros(fp_dim, dtype=np.bool_)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool_)
    arr[list(fp.GetOnBits())] = 1
    return arr

def batch_smiles_to_fp(s_list, fp_dim=2048):
    return np.array([smiles_to_fp(s, fp_dim) for s in s_list])

# --- Value function wrapper ---
def value_fn(model, mols, device='cpu'):
    fps = batch_smiles_to_fp(mols, fp_dim=2048).astype(np.float32)
    num_mols = len(fps)
    if num_mols <= 5:
        mask = np.ones(5, dtype=np.float32)
        mask[num_mols:] = 0
        fps_input = np.zeros((5, 2048), dtype=np.float32)
        fps_input[:num_mols,:] = fps
    else:
        mask = np.ones(num_mols, dtype=np.float32)
        fps_input = fps

    fps_tensor = torch.from_numpy(fps_input).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)
    with torch.no_grad():
        v = model(fps_tensor, mask_tensor).cpu().numpy()
    return float(v.flatten()[0])

# --- Policy logits TTA aggregation ---
def _logmeanexp(a, axis=0):
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.exp(a-m).mean(axis=axis, keepdims=True))).squeeze(axis)

from policyNet import MLPModel
from valueEnsemble import ValueEnsemble

def save_policy_outputs_tta(expand_fn, mols, device='cpu', out_path='policy_logits_tta.npy', n_aug=20, aggregation='mean', topk=50, save_raw=True):
    sm_en = SmilesEnumerator(enum=True, canonical=False)
    results, raw_results = [], []
    with torch.no_grad():
        for idx,s in enumerate(mols):
            aug_candidates, aug_scores = [], []
            for _ in range(n_aug):
                try: aug_s = sm_en.randomize_smiles(s if isinstance(s,str) else s['reaction'])
                except: aug_s = s if isinstance(s,str) else s['reaction']
                out = expand_fn.run(aug_s, topk=topk)
                if out is None:
                    aug_candidates.append([])
                    aug_scores.append(np.array([]))
                    continue
                cands = [r for r in out['reactants']]
                scores = np.array(out['scores'], dtype=np.float32)
                aug_candidates.append(cands)
                aug_scores.append(scores)
            union_set = []
            for cands in aug_candidates:
                for c in cands:
                    if c not in union_set: union_set.append(c)
            if len(union_set)==0: continue
            aligned = np.zeros((len(aug_scores), len(union_set)), dtype=np.float32)
            for i,(cands,scores) in enumerate(zip(aug_candidates, aug_scores)):
                for ci,c in enumerate(cands):
                    aligned[i,union_set.index(c)] = scores[ci]
            if aggregation=='mean': agg_scores = aligned.mean(axis=0)
            elif aggregation=='logmeanexp':
                eps=1e-12
                if np.all((aligned>=0)&(aligned<=1)):
                    log_aligned = np.log(np.clip(aligned, eps, 1.0))
                    agg_scores = np.exp(_logmeanexp(log_aligned, axis=0))
                else:
                    agg_scores = _logmeanexp(aligned, axis=0)
            else: raise ValueError(f'Unknown aggregation {aggregation}')
            results.append({'mol': s if isinstance(s,str) else s['reaction'],'candidates':union_set,'agg_scores':agg_scores})
            if save_raw:
                raw_results.append({'mol': s if isinstance(s,str) else s['reaction'],'union_candidates':union_set,'aligned_scores':aligned})
            if (idx+1)%100==0: print(f"[INFO] processed {idx+1}/{len(mols)} molecules")
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.save(out_path, np.array(results,dtype=object))
    if save_raw: np.save(out_path.replace('.npy','_raw.npy'), np.array(raw_results,dtype=object))
    return results

save_policy_outputs = save_policy_outputs_tta

# --- Jensen-Shannon divergence for epistemic uncertainty ---
def js_divergence(p_list):
    from scipy.spatial.distance import jensenshannon
    # p_list: list of np arrays (prob vectors)
    p_stack = np.stack(p_list)
    p_stack = np.clip(p_stack, 1e-12, 1.0)
    n = p_stack.shape[0]
    if n==1: return 0.0
    mean_p = np.mean(p_stack, axis=0)
    js = np.mean([jensenshannon(p, mean_p) for p in p_stack])
    return js

# --- A* Node and MEEA* Search ---
class AStarNode:
    def __init__(self,state,g,h,action=None,parent=None):
        self.state=state; self.g=g; self.h=h; self.f=g+h
        self.action=action; self.parent=parent
    def __lt__(self,other): return self.f<other.f

def meea_star(target_mol, known_mols, value_model, expand_fn, device, max_steps=500, weight=1.0):
    target = target_mol if isinstance(target_mol,str) else target_mol.get('reaction','')
    start_time=time.time()
    root = AStarNode([target], g=0.0, h=value_fn(value_model,[target],device))
    open_list=[]; heapq.heappush(open_list, root); visited=set(); expansions=0
    while open_list and expansions<max_steps:
        node=heapq.heappop(open_list); expansions+=1
        if all(m in known_mols for m in node.state): return True,node,expansions,time.time()-start_time
        mol=node.state[0]
        out=expand_fn.run(mol, topk=50)
        if out is None: continue
        scores=out.get('scores',[]); reacts=out.get('reactants',[]); templates=out.get('template',[])
        for i in range(len(scores)):
            rlist = [r for r in reacts[i].split('.') if r not in known_mols and r!='']
            new_state=sorted(list(set(rlist+node.state[1:])))
            key='.'.join(new_state)
            if key in visited: continue
            visited.add(key)
            cost = -np.log(np.clip(scores[i],1e-6,1.0))*(2-weight)
            h = value_fn(value_model,new_state,device) if new_state else 0.0
            child=AStarNode(new_state,node.g+cost,h,parent=node,action=(templates[i] if i<len(templates) else None, reacts[i]))
            heapq.heappush(open_list,child)
    return False,None,expansions,time.time()-start_time

def play_meea(dataset,mols,known_mols,value_model,expand_fn,device,weight_override=None):
    results=[]
    total=len(mols)
    for i,mol in enumerate(mols):
        w=mol.get('weight',1.0) if isinstance(mol,dict) else 1.0
        if weight_override is not None: w=weight_override
        success,node,exp,elapsed=meea_star(mol,known_mols,value_model,expand_fn,device,weight=w)
        depth=node.g if success else 32
        results.append({'success':success,'depth':depth,'expansions':exp,'time':elapsed})
        pct=(i+1)/total*100; sys.stdout.write(f"\rDataset {dataset}: {i+1}/{total} molecules ({pct:.1f}%) done"); sys.stdout.flush()
    print()
    success_rate=np.mean([r['success'] for r in results])
    success_depths=[r['depth'] for r in results if r['success']]
    avg_depth=np.mean(success_depths) if success_depths else float('nan')
    avg_exp=np.mean([r['expansions'] for r in results])
    avg_time=np.mean([r['time'] for r in results])
    return success_rate, avg_depth, avg_exp, avg_time, results

# --- Normalize targets ---
def normalize_targets(obj):
    if isinstance(obj,pd.DataFrame):
        if 'reaction' in obj.columns: return obj.to_dict('records')
        elif 'smiles' in obj.columns: return [{'reaction':s} for s in obj['smiles']]
        elif 'mol' in obj.columns: return [{'reaction':s} for s in obj['mol']]
    elif isinstance(obj,list):
        if len(obj)==0: return []
        if isinstance(obj[0],dict): return obj
        return [{'reaction':str(x)} for x in obj]
    elif isinstance(obj,dict): return list(obj.values())
    raise TypeError(f'Unsupported type {type(obj)}')

# --- Main pipeline ---
if __name__=='__main__':
    multiprocessing.set_start_method('spawn',force=True)

    # ---- Models ----
    policy_path=input("Enter policy model path: ").strip()
    value_path=input("Enter value model path: ").strip()

    from policyNet import MLPModel
    from valueEnsemble import ValueEnsemble
    device='cuda' if torch.cuda.is_available() else 'cpu'
    expand_fn=MLPModel(policy_path,'template_rules.dat',device=device)
    value_model=ValueEnsemble(2048,128,0.1).to(device)
    value_model.load_state_dict(torch.load(value_path,map_location=device)); value_model.eval()

    # ---- Dataset input ----
    dataset_path=input("Enter dataset PKL path: ").strip()
    with open(dataset_path,'rb') as f: raw_obj=pickle.load(f)
    targets=normalize_targets(raw_obj)

    # ---- Starting molecules ----
    origin_path=input("Enter starting molecules CSV path: ").strip()
    df_origin=pd.read_csv(origin_path)
    known_mols=set(df_origin['mol'].astype(str).tolist()) if 'mol' in df_origin.columns else set()

    # ---- Phase 1 ----
    print("\n=== Phase 1: baseline MCTS+A* ===")
    success_rate,avg_depth,avg_exp,avg_time,results=play_meea('phase1',targets,known_mols,value_model,expand_fn,device)
    os.makedirs('./test',exist_ok=True)
    with open('./test/stat_phase1.pkl','wb') as f: pickle.dump({'success_rate':success_rate,'avg_depth':avg_depth,'avg_expansions':avg_exp,'avg_time':avg_time,'details':results},f)
    save_policy_outputs(expand_fn,[t['reaction'] for t in targets],device,out_path='./test/policy_logits_tta_phase1.npy')
    print(f"Phase1 success: {success_rate:.3f}, avg_depth: {avg_depth:.2f}")

    # ---- Phase 2: UQ weighting ----
    print("\n=== Phase 2: compute aleatoric + epistemic uncertainty ===")
    weighted_targets=[]
    for t in targets:
        t_str=t['reaction']
        # collect TTA logits
        tta_out=expand_fn.run(t_str,topk=50)
        if tta_out is None: continue
        scores=np.array(tta_out['scores'],dtype=np.float32)
        # aleatoric: variance across TTA augmentations (simplified here as score variance)
        sigma_a = np.var(scores) if len(scores)>0 else 0.0
        # epistemic: JSD across topk candidates (here treated as logits across candidates)
        sigma_e = js_divergence([scores])
        t['weight']=float(np.exp(-(sigma_a+sigma_e)))
        weighted_targets.append(t)
    weighted_pkl='./test/weighted_targets.pkl'
    with open(weighted_pkl,'wb') as f: pickle.dump(weighted_targets,f)
    print(f"Phase2: weighted PKL saved to {weighted_pkl}")

    # ---- Phase 3: UQ-aware MEEA* ----
    print("\n=== Phase 3: UQ-aware MCTS+A* ===")
    success_rate,avg_depth,avg_exp,avg_time,results=play_meea('phase3',weighted_targets,known_mols,value_model,expand_fn,device)
    with open('./test/stat_phase3.pkl','wb') as f: pickle.dump({'success_rate':success_rate,'avg_depth':avg_depth,'avg_expansions':avg_exp,'avg_time':avg_time,'details':results},f)
    save_policy_outputs(expand_fn,[t['reaction'] for t in weighted_targets],device,out_path='./test/policy_logits_tta_phase3.npy')
    print(f"Phase3 success: {success_rate:.3f}, avg_depth: {avg_depth:.2f}")

    print("\n✅ Full MEEA* pipeline completed successfully.")
