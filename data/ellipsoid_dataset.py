#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json, os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple

SS_VOCAB = ['alpha','beta','loop','cdr']
ROLE_VOCAB = ['Non-Paratope','Non-Epitope','Paratope','Epitope']

def _one_hot(i, K):
    x = torch.zeros(K, dtype=torch.float32); 
    if 0 <= i < K: x[i] = 1.0
    return x

def _symmetrize(M): return 0.5*(M+M.transpose(-1,-2))

def _to_tensor3x3(v):
    # accepts 3x3, 6-vector (xx,yx,yy,zx,zy,zz), or flat list len=9
    t = torch.tensor(v, dtype=torch.float32)
    if t.shape == (3,3): 
        return _symmetrize(t)
    if t.numel() == 6:
        out = torch.zeros(3,3, dtype=torch.float32)
        out[0,0]=t[0]; out[1,0]=out[0,1]=t[1]; out[1,1]=t[2]
        out[2,0]=out[0,2]=t[3]; out[2,1]=out[1,2]=t[4]; out[2,2]=t[5]
        return _symmetrize(out)
    if t.numel() == 9:
        return _symmetrize(t.view(3,3))
    raise ValueError("Unrecognized covariance format")

def _invariants(S):
    vals,_ = torch.linalg.eigh(S)
    vals = torch.clamp(vals, min=1e-8)
    tr   = torch.einsum('ii->', S)
    logd = torch.log(vals).sum()
    an   = torch.log(vals[-1]/vals[0])
    return float(tr), float(logd), float(an)

def _first(d:Dict, keys:List[str], default=None):
    for k in keys:
        if k in d: return d[k]
    return default

def _labels_to_onehot(e:Dict, chain2idx:Dict[str,int]):
    ss_raw = str(_first(e, ['ss','SS','secstruct','secondary','secondary_structure'], 'loop')).lower()
    role_raw = _first(e, ['role','label','interface_role'], 'Non-Epitope')
    chain_raw = str(_first(e, ['chain','chain_id','tag'], 'A'))

    ss_idx = SS_VOCAB.index(ss_raw) if ss_raw in SS_VOCAB else 2
    role_idx = ROLE_VOCAB.index(role_raw) if role_raw in ROLE_VOCAB else 1
    if chain_raw not in chain2idx: chain2idx[chain_raw] = len(chain2idx)
    chain_idx = chain2idx[chain_raw]

    oh = torch.cat([
        _one_hot(ss_idx, len(SS_VOCAB)),
        _one_hot(role_idx, len(ROLE_VOCAB)),
        _one_hot(chain_idx, len(chain2idx))
    ], dim=0)
    return ss_idx, role_idx, chain_idx, oh

def _ellipsoid_to_token(e:Dict, chain2idx:Dict[str,int], default_role=None):
    mu = torch.tensor(_first(e, ['mu','center','mean'], [0,0,0]), dtype=torch.float32)
    cov = _first(e, ['cov','Sigma','sigma','covariance'], [[1,0,0],[0,1,0],[0,0,1]])
    Sigma = _to_tensor3x3(cov)
    n = float(_first(e, ['n','count','num','size'], 0))

    # override role if caller provides (e.g., when merging paratope/epitope lists)
    if default_role is not None: e = {**e, 'role': default_role}

    tr, logdet, aniso = _invariants(Sigma)
    ss_idx, role_idx, chain_idx, labels_oh = _labels_to_onehot(e, chain2idx)

    # optional ESM pooling
    esm = _first(e, ['esm','esm_residue','esm_pooled'], None)
    feat = torch.tensor([n, tr, logdet, aniso], dtype=torch.float32)
    if esm is not None and len(esm)>0:
        E = torch.tensor(esm, dtype=torch.float32)
        if E.ndim==1:  # already pooled
            feat = torch.cat([feat, E], dim=0)
        else:
            feat = torch.cat([feat, E.mean(0), E.max(0).values], dim=0)

    token = torch.cat([feat, labels_oh], dim=0)
    return mu, Sigma, n, ss_idx, role_idx, chain_idx, token

class EllipsoidComplexDataset(Dataset):
    """
    Robust loader:
      - accepts JSONL (one record per line) or JSON (list OR dict of records)
      - flexible keys for sets:
          context:  context|non_interface|noninterface|ctx|framework|non_itf|E_ctx
          interface: interface|itf|intf|targets|E_int
          OR split keys: paratope|epitope (will be concatenated with roles set)
    """
    def __init__(self, path:str, jitter_mu_sigma:float=1.0, jitter_logSigma_sigma:float=0.1):
        self.records = self._load_any(path)
        self.jitter_mu_sigma = jitter_mu_sigma
        self.jitter_logSigma_sigma = jitter_logSigma_sigma

        # build chain vocab across the whole dataset for stable dims
        self.chain2idx: Dict[str,int] = {}
        for r in self.records:
            ctx, itf = self._resolve_sets(r)
            for e in ctx + itf:
                _ = _labels_to_onehot(e, self.chain2idx)

        # infer token_dim using the first record (now that chain2idx is fixed)
        tmp = self._make_example(self.records[0])
        self._token_dim = tmp['context_tokens'].shape[-1] if tmp['context_tokens'].numel() else tmp['interface_tokens'].shape[-1]

    @property
    def label_dims(self):
        return {'ss': len(SS_VOCAB), 'role': len(ROLE_VOCAB), 'chain': len(self.chain2idx)}

    @property
    def token_dim(self): return self._token_dim

    def _load_any(self, path: str):
        import json, os
        assert os.path.exists(path), f"File not found: {path}"
        # 1) try regular JSON (list or dict)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return list(data.values())
        except json.JSONDecodeError:
            pass
        # 2) fallback: JSONL (one JSON object per line)
        records = []
        with open(path, "r") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"{path}: line {i} is not valid JSON; if this isn't JSONL, convert it first. {e}"
                    )
        if not records:
            raise ValueError(f"{path} appears empty or not JSON/JSONL.")
        return records

    def _resolve_sets(self, rec:Dict[str,Any]) -> Tuple[List[Dict], List[Dict]]:
        # try direct keys
        ctx = _first(rec, ['context','non_interface','noninterface','ctx','framework','non_itf','E_ctx'], [])
        itf = _first(rec, ['interface','itf','intf','targets','E_int'], [])

        # if split by paratope/epitope, merge and assign roles
        par = rec.get('paratope', [])
        epi = rec.get('epitope', [])
        if (len(par)+len(epi))>0:
            itf = list(par) + list(epi)
            # mark roles if missing
            for e in par:
                e.setdefault('role','Paratope')
            for e in epi:
                e.setdefault('role','Epitope')

        # If still empty, try older naming
        if len(ctx)==0: ctx = rec.get('non_itf_blobs', rec.get('E_context', []))
        if len(itf)==0:
            itf = rec.get('itf_blobs', rec.get('E_target', []))

        if len(ctx)==0 or len(itf)==0:
            # last resort: a single list with role flags
            all_e = rec.get('ellipsoids', [])
            ctx = [e for e in all_e if e.get('role') in ['Non-Paratope','Non-Epitope']]
            itf = [e for e in all_e if e.get('role') in ['Paratope','Epitope']]
        return ctx, itf

    def _make_example(self, rec:Dict[str,Any]) -> Dict[str,torch.Tensor]:
        chain2idx = self.chain2idx
        ctx_list, itf_list = self._resolve_sets(rec)

        ctx_tokens=[]; ctx_mu=[]; ctx_Sigma=[]; ctx_n=[]; ctx_ss=[]; ctx_role=[]; ctx_chain=[]
        for e in ctx_list:
            mu,Sigma,n,ss,role,chain,tok = _ellipsoid_to_token(e, chain2idx)
            ctx_tokens.append(tok); ctx_mu.append(mu); ctx_Sigma.append(Sigma); ctx_n.append(n)
            ctx_ss.append(ss); ctx_role.append(role); ctx_chain.append(chain)

        itf_tokens=[]; itf_mu=[]; itf_Sigma=[]; itf_n=[]; itf_ss=[]; itf_role=[]; itf_chain=[]
        for e in itf_list:
            mu,Sigma,n,ss,role,chain,tok = _ellipsoid_to_token(e, chain2idx)
            itf_tokens.append(tok); itf_mu.append(mu); itf_Sigma.append(Sigma); itf_n.append(n)
            itf_ss.append(ss); itf_role.append(role); itf_chain.append(chain)

        def stack(L, pad_dim=None):
            return (torch.stack(L, dim=0) if len(L)>0 else torch.zeros(0, *(pad_dim or ()), dtype=torch.float32))

        return {
            'context_tokens': stack(ctx_tokens),
            'context_mu': stack(ctx_mu,(3,)),
            'context_Sigma': stack(ctx_Sigma,(3,3)),
            'context_n': (torch.tensor(ctx_n, dtype=torch.float32).unsqueeze(-1) if len(ctx_n)>0 else torch.zeros(0,1)),
            'context_ss': (torch.tensor(ctx_ss, dtype=torch.long) if len(ctx_ss)>0 else torch.zeros(0,dtype=torch.long)),
            'context_role': (torch.tensor(ctx_role, dtype=torch.long) if len(ctx_role)>0 else torch.zeros(0,dtype=torch.long)),
            'context_chain': (torch.tensor(ctx_chain, dtype=torch.long) if len(ctx_chain)>0 else torch.zeros(0,dtype=torch.long)),
            'interface_tokens': stack(itf_tokens),
            'interface_mu': stack(itf_mu,(3,)),
            'interface_Sigma': stack(itf_Sigma,(3,3)),
            'interface_n': (torch.tensor(itf_n, dtype=torch.float32).unsqueeze(-1) if len(itf_n)>0 else torch.zeros(0,1)),
            'interface_ss': (torch.tensor(itf_ss, dtype=torch.long) if len(itf_ss)>0 else torch.zeros(0,dtype=torch.long)),
            'interface_role': (torch.tensor(itf_role, dtype=torch.long) if len(itf_role)>0 else torch.zeros(0,dtype=torch.long)),
            'interface_chain': (torch.tensor(itf_chain, dtype=torch.long) if len(itf_chain)>0 else torch.zeros(0,dtype=torch.long)),
        }

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        ex = self._make_example(self.records[idx])
        # jitter
        if ex['interface_mu'].numel()>0 and self.jitter_mu_sigma>0:
            ex['interface_mu'] = ex['interface_mu'] + torch.randn_like(ex['interface_mu']) * self.jitter_mu_sigma
        if ex['interface_Sigma'].numel()>0 and self.jitter_logSigma_sigma>0:
            vals, vecs = torch.linalg.eigh(ex['interface_Sigma'])
            vals = torch.clamp(vals, min=1e-6)
            logv = torch.log(vals) + torch.randn_like(vals) * self.jitter_logSigma_sigma
            ex['interface_Sigma'] = _symmetrize(vecs @ torch.diag_embed(torch.exp(logv)) @ vecs.transpose(-1,-2))
        return ex

def collate_ellipsoid(batch):
    import torch
    B = len(batch)
    Cmax = max(x['context_tokens'].shape[0] for x in batch)
    Smax = max(x['interface_tokens'].shape[0] for x in batch)
    # ensure at least one slot so src_len/tgt_len > 0
    Cmax = max(Cmax, 1)
    Smax = max(Smax, 1)

    def _tokdim(b):
        dims=[]
        if b['context_tokens'].numel()>0: dims.append(b['context_tokens'].shape[-1])
        if b['interface_tokens'].numel()>0: dims.append(b['interface_tokens'].shape[-1])
        return max(dims) if dims else 0
    token_dim = max(_tokdim(b) for b in batch)

    def pad_feat(T, L, D):
        if T.numel()==0: return torch.zeros(L, D, dtype=torch.float32)
        pad = L - T.shape[0]
        if pad<=0: return T
        return torch.cat([T, torch.zeros(pad, T.shape[-1], dtype=T.dtype)], dim=0)

    def pad_misc(T, L, fill=0.0):
        if T.numel()==0: return torch.full((L,)+T.shape[1:], fill, dtype=torch.float32)
        pad = L - T.shape[0]
        if pad<=0: return T
        return torch.cat([T, torch.full((pad,)+T.shape[1:], fill, dtype=T.dtype)], dim=0)

    def expand_dim(T, D):
        return torch.cat([T, torch.zeros(T.shape[0], D - T.shape[-1])], dim=-1) if T.shape[-1] < D else T

    ctx_tokens = torch.stack([pad_feat(expand_dim(x['context_tokens'], token_dim), Cmax, token_dim) for x in batch], dim=0)
    int_tokens = torch.stack([pad_feat(expand_dim(x['interface_tokens'], token_dim), Smax, token_dim) for x in batch], dim=0)

    out = {
        'context_tokens': ctx_tokens,
        'context_mu': torch.stack([pad_misc(x['context_mu'], Cmax) for x in batch], dim=0),
        'context_Sigma': torch.stack([pad_misc(x['context_Sigma'], Cmax) for x in batch], dim=0),
        'context_n': torch.stack([pad_misc(x['context_n'], Cmax) for x in batch], dim=0),
        'context_ss': torch.stack([pad_misc(x['context_ss'], Cmax, 0).long() for x in batch], dim=0),
        'context_role': torch.stack([pad_misc(x['context_role'], Cmax, 1).long() for x in batch], dim=0),
        'context_chain': torch.stack([pad_misc(x['context_chain'], Cmax, 0).long() for x in batch], dim=0),

        'interface_tokens': int_tokens,
        'interface_mu': torch.stack([pad_misc(x['interface_mu'], Smax) for x in batch], dim=0),
        'interface_Sigma': torch.stack([pad_misc(x['interface_Sigma'], Smax) for x in batch], dim=0),
        'interface_n': torch.stack([pad_misc(x['interface_n'], Smax) for x in batch], dim=0),
        'interface_ss': torch.stack([pad_misc(x['interface_ss'], Smax, 2).long() for x in batch], dim=0),
        'interface_role': torch.stack([pad_misc(x['interface_role'], Smax, 2).long() for x in batch], dim=0),
        'interface_chain': torch.stack([pad_misc(x['interface_chain'], Smax, 0).long() for x in batch], dim=0),

        # masks: True = valid; examples with 0 items become all False
        'context_mask': (torch.arange(Cmax)[None,:].repeat(B,1) <
                         torch.tensor([min(x['context_tokens'].shape[0], Cmax) for x in batch])[:,None]),
        'interface_mask': (torch.arange(Smax)[None,:].repeat(B,1) <
                           torch.tensor([min(x['interface_tokens'].shape[0], Smax) for x in batch])[:,None]),
        'token_dim': token_dim
    }
    return out