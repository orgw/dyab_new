#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lightning interface for Step-1 (Coarse) ellipsoid inpainting with conditional flow matching.

Implements:
- End-state training objective (predict x1 from x_t), following dyAb's fine-grained CFM. 
- Hungarian matching between predicted end-states and ground-truth sets.
- SPD-stable losses in log-SPD space.
- Gaussian–Wishart prior + pairwise Mahalanobis repulsion regularizers.
- Classifier-free guidance via context dropout; Euler sampler with ~10 steps by default.
  (dyAb ablation shows ~10 steps is a good default.)

"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from scipy.optimize import linear_sum_assignment

from .ellipsoid_inpainter_model import (
    EllipsoidInpainterModel, matrix_log_spd, matrix_exp_sym, spd_to_vec_sym, vec_sym_to_sym
)

def pack_labels(ss, role, chain):
    return {'ss': ss, 'role': role, 'chain': chain}

def prior_sample_mu(shape, sigma=20.0, device='cpu'):
    return torch.randn(*shape, device=device) * sigma

def prior_sample_logSigma(shape, device='cpu'):
    # sample diag in log-space; off-diagonals zero (symmetric)
    B,S,_ = shape
    diag = torch.randn(B,S,3, device=device) * 0.2  # log-eigs around 0
    logSigma = torch.zeros(B,S,3,3, device=device)
    logSigma[...,0,0] = diag[...,0]
    logSigma[...,1,1] = diag[...,1]
    logSigma[...,2,2] = diag[...,2]
    return logSigma

def prior_sample_z(shape, mean=2.0, std=1.0, device='cpu'):
    # z = log(1+n); sample around n≈e^{mean}-1
    return torch.randn(*shape, device=device) * std + mean

def pairwise_mahalanobis_repulsion(mu, Sigma, temp=1.0, eps=1e-6):
    """
    mu: (B,S,3), Sigma: (B,S,3,3) SPD
    returns scalar repulsion energy (sum over pairs)
    """
    B,S,_ = mu.shape
    energy = mu.new_zeros(())
    for b in range(B):
        dE = 0.0
        for i in range(S):
            for j in range(i+1, S):
                Sij = Sigma[b,i] + Sigma[b,j]
                diff = (mu[b,i] - mu[b,j]).unsqueeze(0)  # (1,3)
                try:
                    inv = torch.linalg.inv(Sij + eps*torch.eye(3, device=mu.device))
                except RuntimeError:
                    inv = torch.inverse(Sij + eps*torch.eye(3, device=mu.device))
                d2 = (diff @ inv @ diff.transpose(-1,-2)).squeeze()
                dE += torch.exp(-0.5 * d2 / temp)
        energy = energy + dE
    return energy / B

class EllipsoidInpaintingITF(pl.LightningModule):
    def __init__(
        self,
        token_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        label_dims: dict,
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        cf_drop_prob: float = 0.1,
        lam_mu: float = 1.0,
        lam_sigma: float = 0.5,
        lam_z: float = 0.1,
        lam_label: float = 0.2,
        lam_presence: float = 0.05,
        lam_gw: float = 1e-3,
        lam_repulsion: float = 1e-3,
        sampler_steps: int = 10
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = EllipsoidInpainterModel(
            token_dim=token_dim, hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers,
            label_dims=label_dims, use_presence=True
        )

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
        return [opt], [sched]

    # ------------------ core training utils ------------------
    def _match_sets(self, mu_hat, logSigma_hat, z_hat, labels_logits, tgt_mu, tgt_logSigma, tgt_z, tgt_labels, mask):
        """
        Hungarian matching on a per-item basis.
        Cost combines center distance, log-Σ distance, count distance and label CE.
        """
        B,S,_ = mu_hat.shape
        device = mu_hat.device
        idx_pred = []
        idx_tgt = []
        for b in range(B):
            valid = mask[b]  # (S,)
            s = int(valid.sum().item())
            if s == 0:
                idx_pred.append(torch.empty(0, dtype=torch.long, device=device))
                idx_tgt.append(torch.empty(0, dtype=torch.long, device=device))
                continue
            # slice
            mu_p = mu_hat[b,:s]              # (s,3)
            mu_t = tgt_mu[b,:s]
            ls_p = logSigma_hat[b,:s]        # (s,6) packed
            ls_t = spd_to_vec_sym(tgt_logSigma[b,:s])  # (s,6)
            z_p = z_hat[b,:s]                # (s,1)
            z_t = tgt_z[b,:s]

            # label CE per pair (use logits)
            ss_logit, role_logit, chain_logit = labels_logits
            ss_p = ss_logit[b,:s]            # (s,Dss)
            role_p = role_logit[b,:s]
            chain_p = chain_logit[b,:s]
            ss_t = tgt_labels['ss'][b,:s]    # (s,)
            role_t = tgt_labels['role'][b,:s]
            chain_t = tgt_labels['chain'][b,:s]

            # compute pairwise costs
            # center
            mu_cost = torch.cdist(mu_p, mu_t, p=2.0)**2   # (s,s)
            # logSigma L2
            sig_cost = torch.cdist(ls_p, ls_t, p=2.0)**2  # (s,s)
            # z L1
            z_cost = torch.cdist(z_p, z_t, p=1.0)         # (s,s)

            # labels: negative log prob of true for each pair
            def ce_pairwise(logits, tgt):
                # logits: (s,D), tgt: (s,)
                # build (s,s) matrix where [i,j] = CE(pred_i, tgt_j)
                logp = logits.log_softmax(-1)  # (s,D)
                # expand tgt to (s,) -> onehot (s,D) -> (s,D)
                tgt_oh = F.one_hot(tgt, num_classes=logits.shape[-1]).float()  # (s,D)
                # For pairwise, compare every i with every j's onehot
                # compute -sum_k tgt_oh[j,k]*logp[i,k]
                ce = -(logp.unsqueeze(1) * tgt_oh.unsqueeze(0)).sum(-1)  # (s,s)
                return ce

            lab_cost = ce_pairwise(ss_p, ss_t) + ce_pairwise(role_p, role_t) + ce_pairwise(chain_p, chain_t)

            C = self.hparams.lam_mu * mu_cost + self.hparams.lam_sigma * sig_cost + self.hparams.lam_z * z_cost + self.hparams.lam_label * lab_cost
            # Solve Hungarian
            row_ind, col_ind = linear_sum_assignment(C.detach().cpu().numpy())
            idx_pred.append(torch.tensor(row_ind, device=device))
            idx_tgt.append(torch.tensor(col_ind, device=device))
        return idx_pred, idx_tgt

    def _gw_prior_energy(self, mu_hat, Sigma_hat, mu_sigma=50.0, wishart_nu=6.0, wishart_scale=20.0):
        """
        Simple Gaussian prior on centers and Wishart-like regularization on covariance magnitude.
        Not a full log-pdf (constants ignored). Just a stabilizer.
        """
        # mu ~ N(0, mu_sigma^2 I): E = ||mu||^2 / (2*mu_sigma^2)
        E_mu = (mu_hat**2).sum(-1).mean() / (2 * mu_sigma**2)
        # Wishart-style: E = tr(Sigma / sI) - (nu - d - 1) log|Sigma|
        d = 3
        logdet = torch.logdet(Sigma_hat + 1e-6*torch.eye(3, device=Sigma_hat.device)).clamp(min=-50, max=50)
        tr_term = torch.einsum('bnii->bn', Sigma_hat / wishart_scale).mean()
        E_sig = (tr_term - (wishart_nu - d - 1.0) * logdet.mean())
        return E_mu + 1e-3 * E_sig

    # ------------------ Lightning steps ------------------
    def training_step(self, batch, batch_idx):
        ctx_tokens = batch['context_tokens']        # (B,C,D)
        ctx_mu = batch['context_mu']                # (B,C,3)
        ctx_mask = batch['context_mask']            # (B,C) bool

        tgt_mu = batch['interface_mu']              # (B,S,3)
        tgt_Sigma = batch['interface_Sigma']        # (B,S,3,3) SPD
        tgt_logSigma = matrix_log_spd(tgt_Sigma)    # (B,S,3,3)
        tgt_z = torch.log1p(batch['interface_n'].clamp(min=0))  # (B,S,1)
        tgt_labels = {
            'ss': batch['interface_ss'],
            'role': batch['interface_role'],
            'chain': batch['interface_chain']
        }
        S = tgt_mu.shape[1]
        B = tgt_mu.shape[0]

        # Prior samples and interpolation
        mu0 = prior_sample_mu((B,S,3), device=self.device)
        logSigma0 = prior_sample_logSigma((B,S,6), device=self.device)  # as matrix
        logSigma0 = logSigma0  # (B,S,3,3)
        z0 = prior_sample_z((B,S,1), device=self.device)

        t = torch.rand(B,S,1, device=self.device)
        mu_t = (1-t)*mu0 + t*tgt_mu
        logSigma1_vec = spd_to_vec_sym(tgt_logSigma)      # (B,S,6)
        logSigma0_vec = spd_to_vec_sym(logSigma0)         # (B,S,6)
        logSigma_t_vec = (1-t)*logSigma0_vec + t*logSigma1_vec
        z_t = (1-t)*z0 + t*tgt_z

        # Two forwards: conditional and unconditional (context dropout)
        out_c = self.model(ctx_tokens, ctx_mask, ctx_mu, mu_t, logSigma_t_vec, z_t, t, drop_context=False)
        out_u = self.model(ctx_tokens, ctx_mask, ctx_mu, mu_t, logSigma_t_vec, z_t, t, drop_context=True) if self.hparams.cf_drop_prob>0 else None

        # Hungarian matching on cond end-state predictions vs targets
        idx_p, idx_t = self._match_sets(
            mu_hat=out_c['mu_hat'],
            logSigma_hat=out_c['logSigma_hat_vec'],
            z_hat=out_c['z_hat'],
            labels_logits=(out_c['ss_logits'], out_c['role_logits'], out_c['chain_logits']),
            tgt_mu=tgt_mu, tgt_logSigma=tgt_logSigma, tgt_z=tgt_z, tgt_labels=tgt_labels,
            mask=batch['interface_mask']
        )

        # Gather matched pairs and compute losses
        loss_mu = 0.0
        loss_sig = 0.0
        loss_z = 0.0
        loss_lab = 0.0
        loss_presence = 0.0
        E_rep = 0.0
        E_gw = 0.0

        for b in range(B):
            sel_p = idx_p[b]; sel_t = idx_t[b]
            if sel_p.numel() == 0: 
                continue
            mu_ph = out_c['mu_hat'][b][sel_p]                   # (s,3)
            mu_th = tgt_mu[b][sel_t]
            loss_mu = loss_mu + F.mse_loss(mu_ph, mu_th)

            ls_ph = vec_sym_to_sym(out_c['logSigma_hat_vec'][b][sel_p])    # (s,3,3) symmetric
            ls_th = tgt_logSigma[b][sel_t]
            loss_sig = loss_sig + F.mse_loss(ls_ph, ls_th)

            z_ph = out_c['z_hat'][b][sel_p]
            z_th = tgt_z[b][sel_t]
            loss_z = loss_z + F.l1_loss(z_ph, z_th)

            # labels CE
            ss_logits = out_c['ss_logits'][b][sel_p]
            role_logits = out_c['role_logits'][b][sel_p]
            chain_logits = out_c['chain_logits'][b][sel_p]
            loss_lab = loss_lab + F.cross_entropy(ss_logits, tgt_labels['ss'][b][sel_t]) \
                                + F.cross_entropy(role_logits, tgt_labels['role'][b][sel_t]) \
                                + F.cross_entropy(chain_logits, tgt_labels['chain'][b][sel_t])

            # presence (all targets present -> label 1)
            if 'presence_logit' in out_c:
                pres = out_c['presence_logit'][b][sel_p].squeeze(-1)
                loss_presence = loss_presence + F.binary_cross_entropy_with_logits(pres, torch.ones_like(pres))

            # Regularizers on predicted end-states (convert logΣ to Σ)
            Sigma_hat = matrix_exp_sym(ls_ph)  # (s,3,3)
            E_rep = E_rep + pairwise_mahalanobis_repulsion(mu_ph.unsqueeze(0), Sigma_hat.unsqueeze(0))
            E_gw = E_gw + self._gw_prior_energy(mu_ph, Sigma_hat)

        # Normalize
        denom = max(1, B)
        loss_mu = loss_mu/denom
        loss_sig = loss_sig/denom
        loss_z = loss_z/denom
        loss_lab = loss_lab/denom
        loss_presence = loss_presence/denom
        E_rep = E_rep/denom
        E_gw = E_gw/denom

        loss = self.hparams.lam_mu*loss_mu + self.hparams.lam_sigma*loss_sig + \
               self.hparams.lam_z*loss_z + self.hparams.lam_label*loss_lab + \
               self.hparams.lam_presence*loss_presence + \
               self.hparams.lam_repulsion*E_rep + self.hparams.lam_gw*E_gw

        self.log_dict({
            'train/loss': loss,
            'train/loss_mu': loss_mu,
            'train/loss_sig': loss_sig,
            'train/loss_z': loss_z,
            'train/loss_lab': loss_lab,
            'train/E_rep': E_rep,
            'train/E_gw': E_gw
        }, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    @torch.no_grad()
    def sample(self, ctx_tokens, ctx_mask, ctx_mu, S, steps=None, guidance_lambda=2.0):
        """
        Euler sampler over x_t with classifier-free guidance on end-state predictions.
        Returns dict of predicted sets (mu, Sigma, n, labels).
        """
        steps = steps or self.hparams.sampler_steps  # ~10 recommended by dyAb ablation
        B = ctx_tokens.shape[0]
        device = ctx_tokens.device

        # initialize x0
        mu0 = prior_sample_mu((B,S,3), device=device)
        logSigma0 = spd_to_vec_sym(prior_sample_logSigma((B,S,6), device=device))
        z0 = prior_sample_z((B,S,1), device=device)

        # start at t=0 state
        mu_t = mu0.clone()
        logSigma_t = logSigma0.clone()
        z_t = z0.clone()

        for k in range(steps):
            t = torch.full((B,S,1), float(k)/steps, device=device)
            # conditional end-state
            out_c = self.model(ctx_tokens, ctx_mask, ctx_mu, mu_t, logSigma_t, z_t, t, drop_context=False)
            # unconditional
            out_u = self.model(ctx_tokens, ctx_mask, ctx_mu, mu_t, logSigma_t, z_t, t, drop_context=True)
            # cfg end-state
            mu_end = (1+guidance_lambda)*out_c['mu_hat'] - guidance_lambda*out_u['mu_hat']
            ls_end = (1+guidance_lambda)*out_c['logSigma_hat_vec'] - guidance_lambda*out_u['logSigma_hat_vec']
            z_end = (1+guidance_lambda)*out_c['z_hat'] - guidance_lambda*out_u['z_hat']

            # Euler update towards predicted end-state, as in dyAb
            dt = 1.0/steps
            mu_t = mu_t + dt*(mu_end - mu0)
            logSigma_t = logSigma_t + dt*(ls_end - logSigma0)
            z_t = z_t + dt*(z_end - z0)

        # Final predictions at t=1
        t = torch.ones(B,S,1, device=device)
        out = self.model(ctx_tokens, ctx_mask, ctx_mu, mu_t, logSigma_t, z_t, t, drop_context=False)

        # Convert to usable forms
        Sigma = matrix_exp_sym(vec_sym_to_sym(out['logSigma_hat_vec']))
        n = torch.clamp(torch.expm1(out['z_hat']), min=0.0).round().long()
        ss = out['ss_logits'].argmax(-1)
        role = out['role_logits'].argmax(-1)
        chain = out['chain_logits'].argmax(-1)
        return {
            'mu': out['mu_hat'],
            'Sigma': Sigma,
            'n': n,
            'labels': pack_labels(ss, role, chain),
            'presence': torch.sigmoid(out.get('presence_logit', torch.zeros_like(n.float())))
        }