from __future__ import annotations
from typing import Optional, Dict

import torch
import torch.nn as nn

from .ica import InvariantCrossAttention


class EllipsoidTokens(nn.Module):
    """
    Project numeric + categorical ellipsoid features to token embeddings.
    Expected per-ellipsoid features:
      - n (float, count)
      - region (int), cdr_type (int), chain (int), interface (int)
    """
    def __init__(self, d_token: int = 128, vocab: Dict[str, Dict[str, int]] = None):
        super().__init__()
        self.d_token = d_token
        self.vocab = vocab or {
            "region": {"cdr": 0, "loop": 1, "framework": 2},
            "cdr_type": {"None": 0, "H1": 1, "H2": 2, "H3": 3, "L1": 4, "L2": 5, "L3": 6},
            "chain": {"H": 0, "L": 1},
            "interface": {"False": 0, "True": 1},
        }
        self.emb_region = nn.Embedding(len(self.vocab["region"]), d_token // 4)
        self.emb_cdr = nn.Embedding(len(self.vocab["cdr_type"]), d_token // 4)
        self.emb_chain = nn.Embedding(len(self.vocab["chain"]), d_token // 8)
        self.emb_iface = nn.Embedding(len(self.vocab["interface"]), d_token // 8)
        num_cont = d_token - (d_token // 4 + d_token // 4 + d_token // 8 + d_token // 8)
        self.proj_cont = nn.Linear(1, num_cont)

    def forward(self, n: torch.Tensor, feat: Dict[str, torch.Tensor]) -> torch.Tensor:
        # n: (B,K), feat[k]: (B,K)
        r = self.emb_region(feat["region"])  # (B,K,*) 
        c = self.emb_cdr(feat["cdr_type"])   # (B,K,*) 
        chain = self.emb_chain(feat["chain"])# (B,K,*) 
        iface = self.emb_iface(feat["interface"])  # (B,K,*) 
        cont = self.proj_cont(n.unsqueeze(-1))      # (B,K,*) 
        return torch.cat([r, c, chain, iface, cont], dim=-1)


class EllipsoidConditionerMixin(nn.Module):
    """Add ProtComposer-style ellipsoid conditioning to a dyAb update block."""
    def __init__(self, d_model: int, d_ellip: int, n_heads: int = 8, dropout: float = 0.0, cf_guidance_p: float = 0.1):
        super().__init__()
        self.ica = InvariantCrossAttention(d_model=d_model, d_ellip=d_ellip, n_heads=n_heads, dropout=dropout)
        self.cf_guidance_p = cf_guidance_p  # probability to drop condition during training
        self._concat_transformer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)

    def maybe_drop_conditioning(self, ellip_tokens: Optional[torch.Tensor], mu: Optional[torch.Tensor], Sigma: Optional[torch.Tensor]):
        if not self.training or ellip_tokens is None:
            return ellip_tokens, mu, Sigma
        if torch.rand(()) < self.cf_guidance_p:
            return None, None, None
        return ellip_tokens, mu, Sigma

    def concat_and_transform(self, s: torch.Tensor, ellip_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        if ellip_tokens is None:
            return self._concat_transformer(s)
        concat = torch.cat([s, ellip_tokens], dim=1)  # (B, N+K, C)
        out = self._concat_transformer(concat)
        return out[:, : s.shape[1], :]

    def inject(self, s: torch.Tensor, R: torch.Tensor, t: torch.Tensor,
               ellip_tokens: Optional[torch.Tensor], mu: Optional[torch.Tensor], Sigma: Optional[torch.Tensor],
               mask: Optional[torch.Tensor] = None, ellip_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if ellip_tokens is None or mu is None or Sigma is None:
            return s
        delta = self.ica(s, R, t, ellip_tokens, mu, Sigma, mask=mask, ellip_mask=ellip_mask)
        return s + delta
