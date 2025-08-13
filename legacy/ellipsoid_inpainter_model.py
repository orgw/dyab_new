#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EllipsoidInpainterModel (CFM)
- Transformer encoder/decoder over ellipsoid tokens
- End-state heads for (mu, logSigma, z=log(1+n)), labels, presence
- Classifier-free guidance supported by context dropout at forward
- Geometry-aware: query takes time-encoded x_t plus distance-weighted context pooling

Notes:
- Σ is represented in log-SPD space during learning for stability.
- End-state prediction follows dyAb's "fine-grained flow matching" where the model
  predicts x1 given x_t; sampling uses an Euler-style update (see LightningModule).

"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# SPD utilities
# ----------------------------
def _symmetrize(M):
    return 0.5 * (M + M.transpose(-1, -2))

def chol_parameter_to_spd(diag_u, off_u):
    """
    Build SPD via Cholesky L from unconstrained params.
    diag_u: (..., 3)    unconstrained for diagonal of L
    off_u:  (..., 3)    unconstrained for (2,1),(3,1),(3,2)
    Returns L @ L^T shape (...,3,3)
    """
    eps = 1e-4
    L = torch.zeros(*diag_u.shape[:-1], 3, 3, device=diag_u.device, dtype=diag_u.dtype)
    d = F.softplus(diag_u) + eps
    L[...,0,0] = d[...,0]
    L[...,1,1] = d[...,1]
    L[...,2,2] = d[...,2]
    L[...,1,0] = off_u[...,0]
    L[...,2,0] = off_u[...,1]
    L[...,2,1] = off_u[...,2]
    Sigma = L @ L.transpose(-1,-2)
    return _symmetrize(Sigma)

def matrix_log_spd(Sigma):
    """Matrix log of SPD (… ,3,3) -> (… ,3,3)"""
    # eigen-decomp
    vals, vecs = torch.linalg.eigh(Sigma)
    vals = torch.clamp(vals, min=1e-8)
    logvals = torch.log(vals)
    return _symmetrize(vecs @ torch.diag_embed(logvals) @ vecs.transpose(-1,-2))

def matrix_exp_sym(S):
    """Matrix exp of symmetric (… ,3,3) -> SPD (… ,3,3)"""
    vals, vecs = torch.linalg.eigh(S)  # S should be symmetric
    expvals = torch.exp(vals)
    return _symmetrize(vecs @ torch.diag_embed(expvals) @ vecs.transpose(-1,-2))

def spd_to_vec_sym(S):
    """Pack symmetric 3x3 into 6-vector (xx, yx, yy, zx, zy, zz)."""
    return torch.stack([S[...,0,0], S[...,1,0], S[...,1,1],
                        S[...,2,0], S[...,2,1], S[...,2,2]], dim=-1)

def vec_sym_to_sym(v):
    """Unpack 6-vector into symmetric 3x3 matrix."""
    out = torch.zeros(*v.shape[:-1], 3,3, device=v.device, dtype=v.dtype)
    out[...,0,0] = v[...,0]
    out[...,1,0] = v[...,1]; out[...,0,1] = v[...,1]
    out[...,1,1] = v[...,2]
    out[...,2,0] = v[...,3]; out[...,0,2] = v[...,3]
    out[...,2,1] = v[...,4]; out[...,1,2] = v[...,4]
    out[...,2,2] = v[...,5]
    return out

def invariants_from_Sigma(Sigma):
    """Return tr, logdet, anisotropy (log(λ_max/λ_min)) from SPD Sigma."""
    tr = torch.einsum('...ii->...', Sigma)
    vals, _ = torch.linalg.eigh(Sigma)
    vals = torch.clamp(vals, min=1e-8)
    logdet = torch.log(vals).sum(-1)
    aniso = torch.log(vals[..., -1] / vals[..., 0])
    return tr, logdet, aniso

# ----------------------------
# Modules
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # (max_len, dim)

    def forward(self, x, t=None):
        # x: (B,N,D); optional scalar t in [0,1] to be encoded per position
        B,N,D = x.shape
        pe = self.pe[:N].unsqueeze(0).to(x.device)  # (1,N,D)
        out = x + pe
        if t is not None:
            # simple FiLM-style time encoding
            t = t.view(B,N,1)
            out = out + torch.sin(2*math.pi*t) + torch.cos(2*math.pi*t)
        return out

class TokenMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class EllipsoidInpainterModel(nn.Module):
    """
    Set-to-set inpainter with end-state heads for (mu, logSigma, z).
    """
    def __init__(
        self,
        token_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 4,
        label_dims: dict = None,        # {'ss':3,'role':4,'chain':K}
        use_presence: bool = True,
        distance_pool_sigma: float = 8.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_dim = token_dim
        self.label_dims = label_dims or {'ss':3, 'role':4, 'chain':2}
        self.use_presence = use_presence
        self.distance_pool_sigma = distance_pool_sigma

        # Context token encoder
        self.ctx_embed = TokenMLP(token_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.ctx_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Query (x_t) embed: invariants + time + optional pooled context
        # For x_t we compute invariants (n_t, trΣ_t, logdetΣ_t, aniso)
        self.q_embed = TokenMLP(4 + 16, hidden_dim)  # 4 invariants + 16-d time encoding

        # Small time encoder
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 16)
        )

        # Decoder
        dec_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        # Heads: end-state predictions
        self.head_mu = nn.Linear(hidden_dim, 3)
        self.head_logSigma = nn.Linear(hidden_dim, 6)  # symmetric log-Σ packed
        self.head_z = nn.Linear(hidden_dim, 1)         # z = log(1+n)

        self.head_ss = nn.Linear(hidden_dim, self.label_dims['ss'])
        self.head_role = nn.Linear(hidden_dim, self.label_dims['role'])
        self.head_chain = nn.Linear(hidden_dim, self.label_dims['chain'])

        if use_presence:
            self.head_presence = nn.Linear(hidden_dim, 1)

        self.posenc = PositionalEncoding(hidden_dim)

    def geo_pool(self, mu_q, mu_ctx, mem):
        """
        Distance-weighted pooling of context memory for each query.
        mu_q: (B,S,3), mu_ctx: (B,C,3), mem: (B,C,H)
        returns: (B,S,H)
        """
        B,S,_ = mu_q.shape
        C = mu_ctx.shape[1]
        # d^2: (B,S,C)
        d2 = torch.cdist(mu_q, mu_ctx, p=2.0)**2
        w = torch.softmax(-d2 / (2 * (self.distance_pool_sigma**2)), dim=-1)  # (B,S,C)
        pooled = torch.einsum('bsc,bch->bsh', w, mem)
        return pooled

    def forward(
        self,
        ctx_tokens, ctx_mask,
        ctx_mu,
        q_mu_t, q_logSigma_t, q_z_t,
        t,
        drop_context: bool = False
    ):
        """
        Args:
            ctx_tokens: (B,C,token_dim) invariants + one-hot labels + optional ESM
            ctx_mask:   (B,C) boolean, True for valid
            ctx_mu:     (B,C,3)
            q_mu_t:     (B,S,3)
            q_logSigma_t: (B,S,6) packed symmetric
            q_z_t:      (B,S,1)  z = log(1+n)
            t:          (B,S,1)  time in [0,1]
            drop_context: bool, if True zero-out context (CF guidance training)

        Returns dict with end-state heads.
        """
        B,S,_ = q_mu_t.shape

        # Encode context
        ctx_x = self.ctx_embed(ctx_tokens)  # (B,C,H)
        if drop_context:
            ctx_x = torch.zeros_like(ctx_x)
        # Apply mask by zeroing paddings before encoder and using attention mask
        mem = self.ctx_encoder(ctx_x, src_key_padding_mask=~ctx_mask)  # (B,C,H)

        # Query embedding
        # Compute invariants from q_logSigma_t (log Σ → Σ)
        q_Sigma_t = matrix_exp_sym(vec_sym_to_sym(q_logSigma_t))  # (B,S,3,3)
        tr, logdet, aniso = invariants_from_Sigma(q_Sigma_t)      # (B,S)
        n_t = torch.clamp(torch.expm1(q_z_t), min=0.0)            # (B,S,1)
        inv = torch.stack([n_t.squeeze(-1), tr, logdet, aniso], dim=-1)  # (B,S,4)

        t_feat = self.time_mlp(t)  # (B,S,16)
        q_in = torch.cat([inv, t_feat], dim=-1)  # (B,S, 4+16)
        q = self.q_embed(q_in)  # (B,S,H)

        # Add distance-weighted pooled context
        pooled = self.geo_pool(q_mu_t, ctx_mu, mem)  # (B,S,H)
        q = q + pooled

        # Decode
        out = self.decoder(tgt=q, memory=mem, memory_key_padding_mask=~ctx_mask)  # (B,S,H)
        out = self.posenc(out)  # tiny positional bias

        # End-state heads
        mu_hat = self.head_mu(out)                 # (B,S,3)
        logSigma_hat_vec = self.head_logSigma(out) # (B,S,6)
        z_hat = self.head_z(out)                   # (B,S,1)

        # Labels
        ss_logits = self.head_ss(out)              # (B,S,Dss)
        role_logits = self.head_role(out)          # (B,S,Drole)
        chain_logits = self.head_chain(out)        # (B,S,Dchain)

        ret = {
            'mu_hat': mu_hat,
            'logSigma_hat_vec': logSigma_hat_vec,
            'z_hat': z_hat,
            'ss_logits': ss_logits,
            'role_logits': role_logits,
            'chain_logits': chain_logits
        }
        if self.use_presence:
            ret['presence_logit'] = self.head_presence(out)

        return ret