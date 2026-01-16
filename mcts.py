import os
import pickle
import time
import torch
import numpy as np
import pandas as pd
from multiprocessing import Process
from tqdm import tqdm
from policyNet import MLPModel
from valueEnsemble import ValueEnsemble
from rdkit import Chem
from rdkit.Chem import AllChem
from contextlib import contextmanager
import signal

# -----------------------------
# Helpers
# -----------------------------
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

def smiles_to_fp(s, fp_dim=2048):
    mol = Chem.MolFromSmiles(s)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    arr = np.zeros(fp.GetNumBits(), dtype=np.float32)
    arr[list(fp.GetOnBits())] = 1
    return arr

def batch_smiles_to_fp(s_list, fp_dim=2048):
    return np.array([smiles_to_fp(s, fp_dim) for s in s_list], dtype=np.float32)

def prepare_starting_molecules():
    return set(list(pd.read_csv('./prepare_data/origin_dict.csv')['mol']))

def prepare_value(model_f, gpu):
    device = torch.device(f'cuda:{gpu}' if gpu >= 0 else 'cpu')
    model = ValueEnsemble(2048, 128, 0.1).to(device)
    model.load_state_dict(torch.load(model_f, map_location=device))
    model.eval()
    return model

def prepare_expand(model_path, gpu):
    device = torch.device(f'cuda:{gpu}' if gpu >= 0 else 'cpu')
    return MLPModel(model_path, './saved_model/template_rules.dat', device=device)

def value_fn(model, mols, device):
    if len(mols) == 0:
        return np.zeros(0)
    fps = batch_smiles_to_fp(mols)
    fps_tensor = torch.FloatTensor(fps).to(device)
    mask = torch.ones(fps_tensor.shape[0], dtype=torch.float32, device=device)
    with torch.no_grad():
        v = model(fps_tensor.unsqueeze(0), mask.unsqueeze(0)).cpu().numpy()
    return v.flatten()

# -----------------------------
# Node for MCTS
# -----------------------------
class Node:
    def __init__(self, state, h, prior, cost=0, action_mol=None, fmove=0, reaction=None, template=None, parent=None, cpuct=1.5):
        self.state = state
        self.cost = cost
        self.h = h
        self.prior = prior
        self.visited_time = 0
        self.is_expanded = False
        self.template = template
        self.action_mol = action_mol
        self.fmove = fmove
        self.reaction = reaction
        self.parent = parent
        self.cpuct = cpuct
        self.children = []
        self.child_illegal = np.array([])
        if parent:
            self.g = parent.g + cost
            self.depth = parent.depth + 1
            parent.children.append(self)
        else:
            self.g = 0
            self.depth = 0
        self.f = self.g + self.h
        self.f_mean_path = []

    def child_Q(self, min_max_stats):
        child_Qs = []
        for c in self.children:
            if len(c.f_mean_path) == 0:
                child_Qs.append(0.0)
            else:
                child_Qs.append(1 - np.mean(min_max_stats.normalize(c.f_mean_path)))
        return np.array(child_Qs)

    def child_N(self):
        return np.array([c.visited_time for c in self.children])

    def child_U(self):
        N = self.child_N() + 1
        return self.cpuct * np.sqrt(max(1, self.visited_time)) * self.child_p() / N

    def child_p(self):
        return np.array([c.prior for c in self.children])

    def select_child(self, min_max_stats):
        action_score = self.child_Q(min_max_stats) + self.child_U() - self.child_illegal
        return np.argmax(action_score)

# -----------------------------
# MinMaxStats for normalization
# -----------------------------
class MinMaxStats:
    def __init__(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')
    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)
    def normalize(self, value):
        if self.maximum > self.minimum:
            return (np.array(value) - self.minimum) / (self.maximum - self.minimum)
        return value

# -----------------------------
# MCTS Player
# -----------------------------
class MCTS_A:
    def __init__(self, target_mol, known_mols, value_model, expand_fn, device, simulations, cpuct):
        self.target_mol = target_mol
        self.known_mols = known_mols
        self.expand_fn = expand_fn
        self.value_model = value_model
        self.device = device
        self.cpuct = cpuct
        self.visited_policy = {}
        root_value = value_fn(value_model, [target_mol], device)[0]
        self.root = Node([target_mol], root_value, prior=1.0, cpuct=cpuct)
        self.min_max_stats = MinMaxStats()
        self.min_max_stats.update(self.root.f)
        self.iterations = 0

    def select_leaf(self):
        current = self.root
        while True:
            current.visited_time += 1
            if not current.is_expanded:
                return current
            best = current.select_child(self.min_max_stats)
            current = current.children[best]

    def expand(self, node):
        node.is_expanded = True
        mol = node.state[0]
        if mol in self.visited_policy:
            policy = self.visited_policy[mol]
        else:
            policy = self.expand_fn.run(mol, topk=50)
            self.visited_policy[mol] = policy
        if policy is None or len(policy['scores']) == 0:
            return False, None
        priors = np.array([1.0/len(policy['scores'])]*len(policy['scores']))
        node.child_illegal = np.zeros(len(policy['scores']))
        for i in range(len(policy['scores'])):
            reactant = [r for r in policy['reactants'][i].split('.') if r not in self.known_mols]
            reactant = sorted(list(set(reactant)))
            cost = -np.log(np.clip(policy['scores'][i], 1e-3, 1.0))
            h = 0 if len(reactant)==0 else value_fn(self.value_model, reactant, self.device)
            Node(reactant, h, priors[i], cost=cost, action_mol=mol, reaction=policy['reactants'][i] + '>>' + mol,
                 template=policy['template'][i], parent=node, cpuct=self.cpuct)
        return True, node.children[0]

    def update(self, node):
        stat = node.f
        self.min_max_stats.update(stat)
        current = node
        while current:
            current.f_mean_path.append(stat)
            current = current.parent

    def search(self, max_iter):
        success = False
        while self.iterations < max_iter:
            node = self.select_leaf()
            success, _ = self.expand(node)
            self.update(node)
            self.iterations += 1
        return success, node

# -----------------------------
# Phase 1: MCTS + A*
# -----------------------------
def phase1(dataset, mols, value_model, expand_fn, device, simulations, cpuct, save_dir='./test'):
    results = []
    for mol in tqdm(mols, desc=f'Phase1 MCTS ({dataset}) GPU{device}'):
        player = MCTS_A(mol, known_mols, value_model, expand_fn, device, simulations, cpuct)
        success, node = player.search(simulations)
        results.append({
            'mol': mol,
            'success': success,
            'depth': node.depth if node else 0,
            'route': node.state if node else []
        })
    # Save txt & pkl
    txt_file = os.path.join(save_dir, f'stat_baseline_{dataset}.txt')
    with open(txt_file, 'w') as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Success rate: {np.mean([r['success'] for r in results]):.3f}\n")
        f.write(f"Avg depth: {np.mean([r['depth'] for r in results]):.2f}\n")
        f.write(f"Total molecules: {len(mols)}\n")
    pkl_file = os.path.join(save_dir, f'stat_baseline_{dataset}.pkl')
    with open(pkl_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"[INFO] Phase1 results saved: {txt_file}")
    return results

# -----------------------------
# MAIN
# -----------------------------
if __name__ == '__main__':
    os.makedirs('./test', exist_ok=True)
    os.makedirs('./uq_outputs', exist_ok=True)
    # clear previous
    for f in os.listdir('./test'):
        os.remove(os.path.join('./test', f))

    known_mols = prepare_starting_molecules()
    datasets = [f.split('.')[0] for f in os.listdir('./test_dataset') if f.endswith('.pkl')]

    model_path = './saved_model/policy_model.ckpt'
    value_model_path = './saved_model/value_pc.pt'

    devices = list(range(torch.cuda.device_count()))
    print(f"[INFO] Detected GPUs: {['cuda:'+str(d) for d in devices]}")

    simulations = 100
    cpuct = 4.0

    value_models = [prepare_value(value_model_path, gpu) for gpu in devices]
    expand_fns = [prepare_expand(model_path, gpu) for gpu in devices]

    # run datasets
    for dataset in datasets:
        with open(f'./test_dataset/{dataset}.pkl', 'rb') as f:
            mols = pickle.load(f)
        splits = np.array_split(mols, len(devices))
        procs = []
        for i, gpu_mols in enumerate(splits):
            p = Process(target=phase1, args=(dataset, gpu_mols, value_models[i], expand_fns[i], devices[i], simulations, cpuct))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    print("[INFO] All datasets completed.")
