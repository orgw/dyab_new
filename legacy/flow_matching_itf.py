# in models/flow_matching_itf.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .ellipsoid_flow_matcher import EllipsoidFlowMatcher
import wandb
import os
import numpy as np

# ... (cholesky_loss function remains the same) ...
def cholesky_loss(pred_cov_sym, target_cov_sym, eps=1e-8):
    def to_full_matrix(sym_elements):
        B, N, _ = sym_elements.shape
        mat = torch.zeros(B, N, 3, 3, device=sym_elements.device, dtype=sym_elements.dtype)
        mat[..., 0, 0] = sym_elements[..., 0]
        mat[..., 1, 1] = sym_elements[..., 2]
        mat[..., 2, 2] = sym_elements[..., 5]
        mat[..., 1, 0] = mat[..., 0, 1] = sym_elements[..., 1]
        mat[..., 2, 0] = mat[..., 0, 2] = sym_elements[..., 3]
        mat[..., 2, 1] = mat[..., 1, 2] = sym_elements[..., 4]
        return mat
    pred_cov = to_full_matrix(pred_cov_sym)
    target_cov = to_full_matrix(target_cov_sym)
    jitter = torch.eye(3, device=pred_cov.device).unsqueeze(0).unsqueeze(0) * eps
    try:
        L_pred = torch.linalg.cholesky(pred_cov + jitter)
        with torch.no_grad():
            L_target = torch.linalg.cholesky(target_cov + jitter)
        return torch.linalg.matrix_norm(L_pred - L_target, ord='fro')**2
    except torch.linalg.LinAlgError:
        return F.mse_loss(pred_cov_sym, target_cov_sym, reduction='none').sum(dim=-1)

class EllipsoidFlowMatchingITF(pl.LightningModule):
    def __init__(self, token_dim, hidden_dim, num_heads, num_layers, lr, 
                 guidance_dropout_rate=0.1, 
                 loss_weight_mu=1.0,
                 loss_weight_n=1.0,
                 loss_weight_cov=1.0,
                 loss_weight_labels=1.0,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = EllipsoidFlowMatcher(token_dim, hidden_dim, num_heads, num_layers)
        self.null_context = nn.Parameter(torch.randn(1, hidden_dim))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def _common_step(self, batch, batch_idx):
        """
        **UPGRADE:**
        - Uses pre-configured weights to balance the different loss components.
        """
        # ... (most of the method is the same as the previous version) ...
        context_tokens, context_mask = batch["context_tokens"], batch["context_mask"]
        target_tokens, target_mask = batch["interface_tokens"], batch["interface_mask"]
        target_mu = batch["interface_mu"]
        if target_mask.sum() == 0: return None
        context_emb = self.model(context_tokens, context_mask)
        if self.training and torch.rand(1).item() < self.hparams.guidance_dropout_rate:
            context_emb = self.null_context.expand(context_emb.size(0), -1)
        target_n, target_cov_sym, target_labels = target_tokens[..., 0:1], target_tokens[..., 1:7], target_tokens[..., 7:17]
        noise_mu, noise_n, noise_cov_sym, noise_labels = torch.randn_like(target_mu), torch.randn_like(target_n), torch.randn_like(target_cov_sym), torch.randn_like(target_labels)
        t = torch.rand(target_mu.size(0), 1, device=self.device).unsqueeze(-1)
        xt_mu, xt_n, xt_cov_sym, xt_labels = t * target_mu + (1 - t) * noise_mu, t * target_n + (1 - t) * noise_n, t * target_cov_sym + (1-t) * noise_cov_sym, t * target_labels + (1 - t) * noise_labels
        xt_feats = torch.cat([xt_n, xt_cov_sym, xt_labels], dim=-1)
        target_mu_velocity, target_n_velocity, target_cov_velocity = target_mu - noise_mu, target_n - noise_n, target_cov_sym - noise_cov_sym
        pred_feat_vel, pred_mu_vel, pred_label_logits_vel = self.model.vector_field(xt_feats, xt_mu, t.squeeze(-1), context_emb)
        
        # --- LOSS CALCULATION WITH WEIGHTS ---
        loss_mu = F.mse_loss(pred_mu_vel, target_mu_velocity, reduction='none').sum(dim=-1)
        pred_n_vel, pred_cov_vel = pred_feat_vel[..., :1], pred_feat_vel[..., 1:]
        loss_n = F.mse_loss(pred_n_vel, target_n_velocity, reduction='none').sum(dim=-1)
        loss_cov = cholesky_loss(pred_cov_vel, target_cov_velocity)
        loss_labels = F.cross_entropy(pred_label_logits_vel.transpose(1, 2), target_labels.transpose(1, 2), reduction='none')

        combined_loss = (self.hparams.loss_weight_mu * loss_mu + 
                         self.hparams.loss_weight_n * loss_n + 
                         self.hparams.loss_weight_cov * loss_cov + 
                         self.hparams.loss_weight_labels * loss_labels)
        
        masked_loss = (combined_loss * target_mask).sum() / (target_mask.sum() + 1e-8)
        return masked_loss

    # ... (training_step, validation_step, sample, and generate_and_log_visualizations remain the same) ...
    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        if loss is not None: self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        if loss is not None: self.log("valid_loss", loss, prog_bar=True, on_epoch=True)
        return loss
    @torch.no_grad()
    def generate_and_log_visualizations(self, batch, epoch):
        self.eval()
        context_tokens, context_mask = batch["context_tokens"], batch["context_mask"]
        generated_output = self.sample(context_tokens, context_mask, num_ellipsoids=batch["interface_tokens"].shape[1])
        pred_mu = generated_output["pred_mu"]
        points = pred_mu[0].cpu().numpy()
        try:
            if self.logger and hasattr(self.logger.experiment, 'log'):
                self.logger.experiment.log({f"epoch_{epoch}_generated_ellipsoids": wandb.Object3D(points)})
                print(f"Logged generated ellipsoid centers for epoch {epoch} to W&B.")
            else: raise Exception("Wandb logger not available.")
        except Exception as e:
            print(f"Could not log to W&B: {e}. Saving to file instead.")
            save_dir = self.trainer.logger.save_dir if self.trainer.logger else "visualizations"
            os.makedirs(os.path.join(save_dir, "visualizations"), exist_ok=True)
            filepath = os.path.join(save_dir, "visualizations", f"epoch_{epoch}_generated_mu.npy")
            np.save(filepath, points)
            print(f"Saved generated ellipsoid centers to {filepath}")
        self.train()
    @torch.no_grad()
    def sample(self, context_tokens, context_mask, num_ellipsoids=20, num_steps=100, guidance_scale=3.0):
        B, device = context_tokens.size(0), self.device
        cond_context_emb = self.model(context_tokens, context_mask)
        uncond_context_emb = self.null_context.expand(B, -1)
        mu_t = torch.randn(B, num_ellipsoids, 3, device=device)
        feats_t_n_cov, feats_t_labels = torch.randn(B, num_ellipsoids, 7, device=device), torch.randn(B, num_ellipsoids, 10, device=device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t_val = torch.full((B, 1), i * dt, device=device)
            feats_t = torch.cat([feats_t_n_cov, feats_t_labels], dim=-1)
            uncond_feat_vel, uncond_mu_vel, uncond_label_vel = self.model.vector_field(feats_t, mu_t, t_val, uncond_context_emb)
            cond_feat_vel, cond_mu_vel, cond_label_vel = self.model.vector_field(feats_t, mu_t, t_val, cond_context_emb)
            pred_mu_vel = uncond_mu_vel + guidance_scale * (cond_mu_vel - uncond_mu_vel)
            pred_feat_vel = uncond_feat_vel + guidance_scale * (cond_feat_vel - uncond_feat_vel)
            pred_label_vel = uncond_label_vel + guidance_scale * (cond_label_vel - uncond_label_vel)
            mu_t += pred_mu_vel * dt
            feats_t_n_cov += pred_feat_vel * dt
            feats_t_labels += pred_label_vel * dt
        final_feats = torch.cat([feats_t_n_cov, feats_t_labels], dim=-1)
        return {"pred_mu": mu_t, "pred_feats": final_feats}