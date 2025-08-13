# in models/ellipsoid_flow_matcher.py
import torch
import torch.nn as nn
from .se3_flow_vector_field import SE3FlowVectorField

class EllipsoidFlowMatcher(nn.Module):
    """
    SE(3)-equivariant flow matching model for ellipsoid generation.
    """
    def __init__(self, token_dim, hidden_dim, num_heads=4, num_layers=3):
        super().__init__()
        self.token_dim = token_dim
        
        # 1. Context Encoder (re-use from your original model)
        self.input_embedding = nn.Linear(token_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 2. SE(3) Equivariant Vector Field
        # Feature dim = n(1) + cov_sym(6) + labels(10) = 17
        ellipsoid_feature_dim = 1 + 6 + 10 
        self.vector_field = SE3FlowVectorField(
            feature_dim=ellipsoid_feature_dim,
            hidden_dim=hidden_dim,
            context_dim=hidden_dim, # Context from the encoder
            num_egnn_layers=4
        )

    def forward(self, context_tokens, context_mask):
        """Encodes the context. The main logic is in the ITF."""
        embedded_ctx = self.input_embedding(context_tokens)
        encoder_padding_mask = ~context_mask
        encoded_ctx = self.encoder(embedded_ctx, src_key_padding_mask=encoder_padding_mask)
        # We take the mean of the context embeddings for a fixed-size representation
        return encoded_ctx.mean(dim=1)