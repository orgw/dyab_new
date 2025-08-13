# in data/ellipsoid_dataset.py
#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset
from io import StringIO
import pickle
from tqdm import tqdm
from typing import List, Dict

from .pdb_utils import Protein

try:
    import pydssp
    DSSP_SS_MAP = {
        'G': 'alpha', 'H': 'alpha', 'I': 'alpha',
        'E': 'beta', 'B': 'beta',
        'T': 'loop', 'S': 'loop', 'C': 'loop',
        '-': 'loop',
    }
except ImportError:
    pydssp = None
    DSSP_SS_MAP = {}

class Ellipsoid:
    def __init__(self, mu, cov, n, ss_type, role, chain_tag):
        self.mu, self.cov, self.n, self.ss_type, self.role, self.chain_tag = mu, cov, n, ss_type, role, chain_tag

class EllipsoidComplexDataset(Dataset):
    def __init__(self, records, pdb_base_path, cache_path=None, distance_cutoff=4.5, radius_thresh=4.5, size_thresh=3):
        super().__init__()
        self.records, self.pdb_base_path = records, pdb_base_path
        self.cache_path = cache_path
        self.distance_cutoff, self.radius_thresh, self.size_thresh = distance_cutoff, radius_thresh, size_thresh
        self.ss_map, self.role_map, self.chain_map = {'alpha': 0, 'beta': 1, 'loop': 2, 'CDR': 2}, {'Paratope': 0, 'Epitope': 1, 'Framework': 2, 'Non-Epitope': 3}, {'VH': 0, 'VL': 1, 'Antigen': 2}
        self.token_dim = 1 + 6 + 3 + 4 + 3 # 17
        self.inv_ss_map = {0: 'alpha', 1: 'beta', 2: 'loop'}
        self.caching_enabled = self.cache_path is not None
        if self.caching_enabled:
            os.makedirs(self.cache_path, exist_ok=True)
            self._pre_process_and_cache()
    def _pre_process_and_cache(self):
        self.cached_file_paths = []
        for idx in tqdm(range(len(self.records)), desc="Processing and Caching Data"):
            file_path = os.path.join(self.cache_path, f"data_{idx}.pkl")
            self.cached_file_paths.append(file_path)
            if os.path.exists(file_path): continue
            processed_item = self._process_item(idx)
            with open(file_path, 'wb') as f: pickle.dump(processed_item, f)
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        if self.caching_enabled:
            with open(self.cached_file_paths[idx], 'rb') as f: return pickle.load(f)
        else: return self._process_item(idx)
    def _process_item(self, idx):
        rec = self.records[idx]
        pdb_path = rec.get("pdb_data_path")
        if not os.path.isabs(pdb_path): pdb_path = os.path.join(self.pdb_base_path, pdb_path)
        if not pdb_path or not os.path.exists(pdb_path): return self._get_empty_item()
        with open(pdb_path, "r") as f: pdb_str_raw = f.read()
        pdb_str = self._sanitize_pdb(pdb_str_raw)
        all_ellipsoids = self._extract_ellipsoids(pdb_str, rec)
        if not all_ellipsoids: return self._get_empty_item()
        context_ellipsoids = [e for e in all_ellipsoids if e.role not in ["Paratope", "Epitope"]]
        interface_ellipsoids = [e for e in all_ellipsoids if e.role in ["Paratope", "Epitope"]]
        return {"context_tokens": self._tokenize_ellipsoids(context_ellipsoids), "interface_tokens": self._tokenize_ellipsoids(interface_ellipsoids),
                "context_mu": torch.from_numpy(np.array([e.mu for e in context_ellipsoids])).float() if context_ellipsoids else torch.empty(0, 3),
                "interface_mu": torch.from_numpy(np.array([e.mu for e in interface_ellipsoids])).float() if interface_ellipsoids else torch.empty(0, 3)}

    def _get_empty_item(self):
        return {"context_tokens": torch.empty(0, self.token_dim), "interface_tokens": torch.empty(0, self.token_dim),
                "context_mu": torch.empty(0, 3), "interface_mu": torch.empty(0, 3)}

    def _tokenize_ellipsoids(self, ellipsoids):
        if not ellipsoids:
            return torch.empty(0, self.token_dim)
        tokens = []
        for e in ellipsoids:
            n_feature = torch.log(torch.tensor([e.n], dtype=torch.float32) + 1e-6)
            if e.cov is not None:
                trace = e.cov[0,0] + e.cov[1,1] + e.cov[2,2]
                norm_cov = e.cov / (trace + 1e-6)
                cov_sym = torch.tensor([
                    norm_cov[0,0], norm_cov[1,0], norm_cov[1,1],
                    norm_cov[2,0], norm_cov[2,1], norm_cov[2,2]
                ], dtype=torch.float32)
            else:
                cov_sym = torch.zeros(6)
            ss_vec = F.one_hot(torch.tensor(self.ss_map.get(e.ss_type, 2)), num_classes=3)
            role_vec = F.one_hot(torch.tensor(self.role_map.get(e.role, 3)), num_classes=4)
            chain_vec = F.one_hot(torch.tensor(self.chain_map.get(e.chain_tag, 2)), num_classes=3)
            token = torch.cat([n_feature, cov_sym, ss_vec.float(), role_vec.float(), chain_vec.float()])
            tokens.append(token)
        return torch.stack(tokens)

    def _extract_ellipsoids(self, pdb_str, rec):
        h_id, l_id, ag_ids = rec.get("heavy_chain"), rec.get("light_chain"), rec.get("antigen_chains", [])
        if not all([h_id, l_id, ag_ids]): return []
        protein = self._from_pdb_string_safe(pdb_str)
        if not protein: return []
        heavy, light = protein.get_chain(h_id), protein.get_chain(l_id)
        antigen, _ = self._load_first_valid_antigen(protein, ag_ids)
        if not all([heavy, light, antigen]): return []
        pH, pL, eAg = self._calculate_interface_residues(heavy, light, antigen)
        dssp_H, dssp_L, dssp_Ag = self._get_secondary_structure(heavy), self._get_secondary_structure(light), self._get_secondary_structure(antigen)
        ell_H, ell_L, ell_Ag = self._segment_and_label(heavy, dssp_H, 'H', pH), self._segment_and_label(light, dssp_L, 'L', pL), self._segment_and_label(antigen, dssp_Ag, 'Ag', eAg)
        return ell_H + ell_L + ell_Ag
    def _from_pdb_string_safe(self, pdb_string, chain_id=None):
        try:
            with StringIO(pdb_string) as pdb_file: return Protein.from_pdb(pdb_file).get_chain(chain_id) if chain_id else Protein.from_pdb(pdb_file)
        except Exception: return None
    def _sanitize_pdb(self, pdb_text, altloc_preference=(" ", "A")):
        return "\n".join([line for line in pdb_text.splitlines() if line.startswith("ATOM  ") and line[16] in altloc_preference])
    def _get_coords_and_res_indices(self, chain_obj):
        if not hasattr(chain_obj, 'residues'): return np.array([]), []
        coords, indices = [], []
        for res in chain_obj:
            if 'CA' in res.coordinate: coords.append(res.get_coord('CA')); indices.append(res.get_id()[0])
        return np.array(coords), indices
    def _get_all_atom_coords(self, residue_obj):
        return np.array(list(residue_obj.coordinate.values())) if hasattr(residue_obj, 'coordinate') and residue_obj.coordinate else np.array([])
    def _calculate_interface_residues(self, heavy_obj, light_obj, antigen_obj):
        if not all([heavy_obj, light_obj, antigen_obj]): return set(), set(), set()
        h_residues, l_residues, ag_residues = list(heavy_obj), list(light_obj), list(antigen_obj)
        ab_residues, num_h_res = h_residues + l_residues, len(h_residues)
        paratope_H, paratope_L, epitope = set(), set(), set()
        ab_atoms_list = [self._get_all_atom_coords(res) for res in ab_residues]
        ag_atoms_list = [self._get_all_atom_coords(res) for res in ag_residues]
        for i, ab_res in enumerate(ab_residues):
            if not ab_atoms_list[i].size: continue
            for j, ag_res in enumerate(ag_residues):
                if not ag_atoms_list[j].size: continue
                if cdist(ab_atoms_list[i], ag_atoms_list[j]).min() < self.distance_cutoff:
                    epitope.add(ag_res.get_id()[0])
                    (paratope_H if i < num_h_res else paratope_L).add(ab_res.get_id()[0])
        return paratope_H, paratope_L, epitope
    def _get_secondary_structure(self, protein_obj):
        if pydssp is None or not hasattr(protein_obj, 'residues'): return np.full(len(protein_obj.residues) if hasattr(protein_obj, 'residues') else 0, 2, dtype=int)
        coords_list, valid_indices = [], []
        for i, res in enumerate(protein_obj):
            if all(k in res.coordinate for k in ['N', 'CA', 'C', 'O']):
                coords_list.append([res.get_coord(atom) for atom in ['N', 'CA', 'C', 'O']])
                valid_indices.append(i)
        if not coords_list: return np.full(len(protein_obj.residues), 2, dtype=int)
        try:
            dssp_numeric = np.array([self.ss_map.get(DSSP_SS_MAP.get(c, 'loop'), 2) for c in pydssp.assign(np.array(coords_list))], dtype=int)
            full_dssp = np.full(len(protein_obj.residues), 2, dtype=int)
            full_dssp[valid_indices] = dssp_numeric
            return full_dssp
        except Exception: return np.full(len(protein_obj.residues), 2, dtype=int)
    def _segment_and_label(self, protein_obj, dssp_indices, chain_type, interface_residues):
        if not hasattr(protein_obj, 'residues') or len(dssp_indices) != len(protein_obj.residues): return []
        ca_pos, res_indices = self._get_coords_and_res_indices(protein_obj)
        if len(ca_pos) < 2: return []
        distmat, G, loop_code = cdist(ca_pos, ca_pos), nx.Graph(), self.ss_map['loop']
        for i, j in np.argwhere(np.triu(distmat < self.radius_thresh, k=1)):
            if dssp_indices[i] == dssp_indices[j] and dssp_indices[i] != loop_code: G.add_edge(i, j)
        for i in range(len(res_indices) - 1):
            if dssp_indices[i] == loop_code and dssp_indices[i+1] == loop_code and abs(res_indices[i] - res_indices[i+1]) == 1: G.add_edge(i, i+1)
        ellipsoids = []
        for component in nx.connected_components(G):
            if len(component) < self.size_thresh: continue
            blob_pos = ca_pos[list(component)]
            covar = np.cov(blob_pos.T) + 1e-6 * np.identity(3) if blob_pos.shape[0] > 1 else np.identity(3) * 1e-9
            ss_type = self.inv_ss_map.get(np.bincount(dssp_indices[list(component)]).argmax(), 'loop')
            is_interface = any(res_indices[i] in interface_residues for i in component)
            role = ('Paratope' if is_interface else 'Framework') if chain_type in ['H','L'] else ('Epitope' if is_interface else 'Non-Epitope')
            chain_tag = "Antigen" if chain_type == 'Ag' else ('VH' if chain_type == 'H' else 'VL')
            ellipsoids.append(Ellipsoid(blob_pos.mean(0), covar, len(component), ss_type, role, chain_tag))
        return ellipsoids
    def _load_first_valid_antigen(self, protein_obj, antigen_chain_ids):
        for ag_id in antigen_chain_ids:
            if (ag := protein_obj.get_chain(ag_id)): return ag, ag_id
        return None, None

# --- RENAMED THIS FUNCTION ---
def ellipsoid_collate_fn(batch: List[Dict[str,torch.Tensor]]):
    # ... (the rest of the function remains exactly the same) ...
    from torch.nn.utils.rnn import pad_sequence
    batch = [item for item in batch if item and (item['context_tokens'].nelement() > 0 or item['interface_tokens'].nelement() > 0)]
    if not batch: 
        return {
            "context_tokens": torch.empty(0, 17), "context_mu": torch.empty(0, 3), "context_mask": torch.empty(0, 0, dtype=torch.bool),
            "interface_tokens": torch.empty(0, 17), "interface_mu": torch.empty(0, 3), "interface_mask": torch.empty(0, 0, dtype=torch.bool),
        }
    ctx_tokens = [item['context_tokens'] for item in batch]
    ctx_mu = [item['context_mu'] for item in batch]
    int_tokens = [item['interface_tokens'] for item in batch]
    int_mu = [item['interface_mu'] for item in batch]

    ctx_tokens_padded = pad_sequence(ctx_tokens, batch_first=True)
    ctx_mu_padded = pad_sequence(ctx_mu, batch_first=True)
    int_tokens_padded = pad_sequence(int_tokens, batch_first=True)
    int_mu_padded = pad_sequence(int_mu, batch_first=True)

    ctx_mask = torch.arange(ctx_tokens_padded.shape[1])[None, :] < torch.tensor([len(t) for t in ctx_tokens])[:, None]
    int_mask = torch.arange(int_tokens_padded.shape[1])[None, :] < torch.tensor([len(t) for t in int_tokens])[:, None]

    return {
        "context_tokens": ctx_tokens_padded, "context_mu": ctx_mu_padded, "context_mask": ctx_mask,
        "interface_tokens": int_tokens_padded, "interface_mu": int_mu_padded, "interface_mask": int_mask,
    }