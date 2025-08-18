import sys, os
# force Python to resolve "conditioning" and "models" from ./dyab/*
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dyab"))
import argparse, torch
from conditioning import extract_antibody_ellipsoids          # <- from dyab/conditioning/*
from models import EllipsoidTokens, InvariantCrossAttention   # <- from dyab/models/*

p = argparse.ArgumentParser()
p.add_argument("--pdb", required=True)
p.add_argument("--heavy", nargs="*", default=None)
p.add_argument("--light", nargs="*", default=None)
p.add_argument("--antigen", nargs="*", default=None)
args = p.parse_args()

# 1) Ellipsoids (CDRs + loops; interface tags)
es = extract_antibody_ellipsoids(args.pdb, heavy_ids=args.heavy, light_ids=args.light, antigen_ids=args.antigen)
mu, Sigma, n, feat, vocab = es.to_torch()               # (K,3), (K,3,3), (K,), dict[(K,)], vocab
print("K ellipsoids:", mu.shape[0])

# 2) Tokens
K = mu.shape[0]
etok = EllipsoidTokens(d_token=128, vocab=vocab)
ellip_tokens = etok(n[None], {k: v[None] for k,v in feat.items()})  # (1,K,128)

# 3) Dummy residue features/frames (as dyAb would provide)
B, N, C = 1, 128, 96
s = torch.randn(B, N, C)
R = torch.eye(3)[None,None].repeat(B, N, 1, 1)          # (B,N,3,3)
t = torch.zeros(B, N, 3)                                # (B,N,3)

# 4) Invariant Cross Attention (ProtComposer-style)
ica = InvariantCrossAttention(d_model=C, d_ellip=ellip_tokens.shape[-1], n_heads=8)
out = ica(s, R, t, ellip_tokens, mu[None], Sigma[None]) # (B,N,C)
print("ICA OK, out shape:", out.shape)