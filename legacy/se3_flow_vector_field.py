# in models/se3_flow_vector_field.py
import torch
import torch.nn as nn
from egnn_pytorch import EGNN

class SE3FlowVectorField(nn.Module):
    """
    An SE(3)-equivariant vector field for flow matching.
    Predicts the velocity vector for a set of ellipsoids at time t,
    conditioned on a context embedding.
    
    **UPGRADES:**
    - Predicts separate outputs for features (n, cov), coordinates (mu), and labels.
    - Covariance prediction is structured for a Cholesky-based SPD loss.
    """
    def __init__(self, feature_dim, hidden_dim, context_dim, num_egnn_layers=4):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.label_dim = 10 # ss(3) + role(4) + chain(3)

        # Input projection to match EGNN's dimension
        self.input_proj = nn.Linear(feature_dim + 1 + context_dim, hidden_dim)

        # The core equivariant network
        self.egnn_layers = nn.ModuleList([
            EGNN(dim=hidden_dim, m_dim=hidden_dim) for _ in range(num_egnn_layers)
        ])
        
        # --- NEW: Separate output heads for each component ---
        # Predicts a change in coordinates (3)
        self.coord_head = nn.Linear(hidden_dim, 3)
        # Predicts a change in scalar (n) and covariance features
        self.feature_head = nn.Linear(hidden_dim, 1 + 6) # n (1) + cov_sym (6)
        # Predicts logits for the labels
        self.label_head = nn.Linear(hidden_dim, self.label_dim)


    def forward(self, feats, coords, t, context_emb):
        """
        Args:
            feats (Tensor): Features of the ellipsoids at time t. Shape (B, N, F).
            coords (Tensor): Coordinates (mu) of ellipsoids at time t. Shape (B, N, 3).
            t (Tensor): Time step. Shape (B, 1).
            context_emb (Tensor): Encoded context. Shape (B, C).

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Predicted velocities for 
                                           (features, coordinates, label_logits).
        """
        B, N, _ = feats.shape
        
        # Broadcast time and context to each ellipsoid
        t_emb = t.unsqueeze(1).repeat(1, N, 1)
        ctx_emb = context_emb.unsqueeze(1).repeat(1, N, 1)
        
        # Combine all inputs
        combined_feats = torch.cat([feats, t_emb, ctx_emb], dim=-1)
        
        # Project to hidden dimension
        h = self.input_proj(combined_feats)
        
        # Pass through EGNN layers
        for egnn in self.egnn_layers:
            h, coords = egnn(h, coords)
            
        # --- NEW: Predict output from separate heads ---
        pred_feat_velocity = self.feature_head(h)
        pred_coord_velocity = self.coord_head(h)
        pred_label_logits = self.label_head(h)
        
        return pred_feat_velocity, pred_coord_velocity, pred_label_logits