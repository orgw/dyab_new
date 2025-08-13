from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Encode 3D relative positions; fixed dim via a learnable mix layer."""
    def __init__(self, dim: int = 192, max_log_freq: float = 7.0):
        super().__init__()
        assert dim % 6 == 0, "dim must be multiple of 6"
        self.dim = dim
        self.num_freq = dim // 6
        self.register_buffer("freqs", 2 ** torch.linspace(0, max_log_freq, steps=self.num_freq))
        self.mix = nn.Linear(6 * self.num_freq, 6 * self.num_freq, bias=False)
        with torch.no_grad():
            nn.init.eye_(self.mix.weight)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        # r: (B,N,K,3)
        freqs = self.freqs.to(r.device)
        x = r[..., None] * freqs  # (B,N,K,3,F)
        sin = torch.sin(x); cos = torch.cos(x)
        feat = torch.cat([sin, cos], dim=-2).flatten(-2)  # (B,N,K, 6*F)
        return self.mix(feat)


class InvariantCrossAttention(nn.Module):
    """ProtComposer-style invariant cross attention (residues → ellipsoids)."""
    def __init__(self, d_model: int, d_ellip: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_ellip = d_ellip
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        self.posenc = SinusoidalPositionalEncoding(dim=d_model)
        self.lin_cov = nn.Linear(9, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(2 * d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model + d_ellip, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        if d_ellip > 0:
            enc_layer = nn.TransformerEncoderLayer(d_model=d_ellip, nhead=max(1, n_heads // 2), batch_first=True)
            self.ellipsoid_encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        else:
            self.ellipsoid_encoder = None

    @staticmethod
    def _to_local(mu: torch.Tensor, Sigma: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # mu: (B,K,3), Sigma: (B,K,3,3), R: (B,N,3,3), t: (B,N,3)
        R_T = R.transpose(-1, -2)                     # (B,N,3,3)
        diff = (mu[:, None, :, :] - t[:, :, None, :])[..., None]  # (B,N,K,3,1)
        r = torch.matmul(R_T[:, :, None, :, :], diff).squeeze(-1)  # (B,N,K,3)
        C = torch.matmul(R_T[:, :, None, :, :], torch.matmul(Sigma[:, None, :, :, :], R[:, :, None, :, :]))  # (B,N,K,3,3)
        return r, C

    def forward(
        self,
        s: torch.Tensor,            # (B,N,d_model)
        R: torch.Tensor,            # (B,N,3,3)
        t: torch.Tensor,            # (B,N,3)
        ellip_tokens: Optional[torch.Tensor], # (B,K,d_ellip)
        mu: torch.Tensor,           # (B,K,3)
        Sigma: torch.Tensor,        # (B,K,3,3)
        mask: Optional[torch.Tensor] = None,       # (B,N)
        ellip_mask: Optional[torch.Tensor] = None  # (B,K)
    ) -> torch.Tensor:
        B, N, C = s.shape
        K = mu.shape[1]

        if self.ellipsoid_encoder is not None and ellip_tokens is not None:
            ellip_tokens = self.ellipsoid_encoder(ellip_tokens)

        r_local, C_local = self._to_local(mu, Sigma, R, t)
        pos = self.posenc(r_local)                  # (B,N,K,d_model)
        cov = self.lin_cov(C_local.flatten(-2))     # (B,N,K,d_model)

        K_mat = self.k_proj(torch.cat([pos, cov], dim=-1))  # (B,N,K,d_model)

        if ellip_tokens is None:
            e = torch.zeros(B, K, self.d_ellip, device=s.device, dtype=s.dtype)
        else:
            e = ellip_tokens
        e = e[:, None, :, :].expand(B, N, K, -1)   # (B,N,K,d_ellip)
        V_mat = self.v_proj(torch.cat([pos, e], dim=-1))    # (B,N,K,d_model)

        Q = self.q_proj(s)                                      # (B,N,d_model)
        dh = C // self.n_heads
        Qh = Q.view(B, N, self.n_heads, dh).permute(0, 2, 1, 3)       # (B,h,N,dh)
        Kh = K_mat.view(B, N, K, self.n_heads, dh).permute(0, 3, 1, 2, 4)  # (B,h,N,K,dh)
        Vh = V_mat.view(B, N, K, self.n_heads, dh).permute(0, 3, 1, 2, 4)  # (B,h,N,K,dh)

        scores = torch.einsum("bhnd,bhnkd->bhnk", Qh, Kh) / math.sqrt(dh)  # (B,h,N,K)
        if ellip_mask is not None:
            scores = scores + (~ellip_mask.bool()).to(scores.dtype)[:, None, None, :] * -1e4
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhnk,bhnkd->bhnd", attn, Vh)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        return self.dropout(self.o_proj(out))
