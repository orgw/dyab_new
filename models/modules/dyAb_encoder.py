#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from .dyAb_egnn import AM_E_GCL


# class dyAbEncoder(nn.Module):

#     def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_channel, channel_nf,
#                  radial_nf, in_edge_nf=0, act_fn=nn.SiLU(), n_layers=4,
#                  residual=True, dropout=0.1, dense=False, init=False):
#         super().__init__()
#         '''
#         :param in_node_nf: Number of features for 'h' at the input
#         :param hidden_nf: Number of hidden features
#         :param out_node_nf: Number of features for 'h' at the output
#         :param n_channel: Number of channels of coordinates
#         :param in_edge_nf: Number of features for the edge features
#         :param act_fn: Non-linearity
#         :param n_layers: Number of layer for the EGNN
#         :param residual: Use residual connections, we recommend not changing this one
#         :param dropout: probability of dropout
#         :param dense: if dense, then context states will be concatenated for all layers,
#                       coordination will be averaged
#         '''
#         self.hidden_nf = hidden_nf
#         self.n_layers = n_layers

#         self.dropout = nn.Dropout(dropout)

#         self.linear_in = nn.Linear(in_node_nf, self.hidden_nf)

#         self.dense = dense
#         if dense:
#             self.linear_out = nn.Linear(self.hidden_nf * (n_layers + 1), out_node_nf)
#         else:
#             self.linear_out = nn.Linear(self.hidden_nf, out_node_nf)

#         for i in range(0, n_layers):
#             self.add_module(f'ctx_gcl_{i}', AM_E_GCL(
#                 self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf, radial_nf,
#                 edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, dropout=dropout, init=init
#             ))
#             self.add_module(f'inter_gcl_{i}', AM_E_GCL(
#                 self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf, radial_nf,
#                 edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, dropout=dropout, init=init
#             ))
#         self.out_layer = AM_E_GCL(
#             self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf,
#             radial_nf, edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, init=init
#         )
    
#     def forward(self, h, x, ctx_edges, inter_mask, inter_x, inter_edges, update_mask, inter_update_mask, channel_attr, channel_weights, ctx_edge_attr=None, H_sv=None, ctx_H_se=None, int_H_se=None):
#         h = self.linear_in(h)
#         h = self.dropout(h)
#         inter_h = h[inter_mask]
#         inter_channel_attr = channel_attr[inter_mask]
#         inter_channel_weights = channel_weights[inter_mask]

#         ctx_states, ctx_coords, inter_coords = [], [], []
#         for i in range(0, self.n_layers):
#             h, x = self._modules[f'ctx_gcl_{i}'](
#                 h, ctx_edges, x, channel_attr, channel_weights,
#                 edge_attr=ctx_edge_attr,
#                 H_sv=H_sv, H_se=ctx_H_se)
#             # synchronization of the shadow paratope (native -> shadow)
#             inter_h = inter_h.clone()
#             inter_h[inter_update_mask] = h[update_mask]
#             inter_h, inter_x = self._modules[f'inter_gcl_{i}'](
#                 inter_h, inter_edges, inter_x, inter_channel_attr, inter_channel_weights,
#                 H_sv=H_sv, H_se=int_H_se
#             )
#             # synchronization of the shadow paratope (shadow -> native)
#             h = h.clone()
#             h[inter_mask] = inter_h
#             ctx_states.append(h)
#             ctx_coords.append(x)
#             inter_coords.append(inter_x)

#         h, x = self.out_layer(
#             h, ctx_edges, x, channel_attr, channel_weights,
#             edge_attr=ctx_edge_attr, H_sv=H_sv, H_se=ctx_H_se)
#         ctx_states.append(h)
#         ctx_coords.append(x)
#         if self.dense:
#             h = torch.cat(ctx_states, dim=-1)
#             x = torch.mean(torch.stack(ctx_coords), dim=0)
#             inter_x = torch.mean(torch.stack(inter_coords), dim=0)
#         h = self.dropout(h)
#         h = self.linear_out(h)
#         return h, x, inter_x

class dyAbEncoder(nn.Module):

    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_channel, channel_nf,
                 radial_nf, in_edge_nf=0, act_fn=nn.SiLU(), n_layers=4,
                 residual=True, dropout=0.1, dense=False, init=False,
                 d_ellip=128, n_heads=8):
                 
        super().__init__()
        '''
        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param n_channel: Number of channels of coordinates
        :param in_edge_nf: Number of features for the edge features
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param dropout: probability of dropout
        :param dense: if dense, then context states will be concatenated for all layers,
                      coordination will be averaged
        '''
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers

        self.dropout = nn.Dropout(dropout)

        self.linear_in = nn.Linear(in_node_nf, self.hidden_nf)

        self.dense = dense
        if dense:
            self.linear_out = nn.Linear(self.hidden_nf * (n_layers + 1), out_node_nf)
        else:
            self.linear_out = nn.Linear(self.hidden_nf, out_node_nf)

        for i in range(0, n_layers):
            self.add_module(f'ctx_gcl_{i}', AM_E_GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf, radial_nf,
                edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, dropout=dropout, init=init
            ))
            self.add_module(f'inter_gcl_{i}', AM_E_GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf, radial_nf,
                edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, dropout=dropout, init=init
            ))
        self.out_layer = AM_E_GCL(
            self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf,
            radial_nf, edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, init=init
        )
            # Initialize the EllipsoidConditionerMixin
        self.ellipsoid_conditioner = EllipsoidConditionerMixin(
            d_model=hidden_nf, d_ellip=d_ellip, n_heads=n_heads, dropout=dropout
        )
        
    def forward(self, h, x, ctx_edges, inter_mask, inter_x, inter_edges, update_mask, 
                inter_update_mask, channel_attr, channel_weights, 
                # Add ellipsoids to the forward pass
                ellipsoids=None,
                ctx_edge_attr=None, H_sv=None, ctx_H_se=None, int_H_se=None):
        h = self.linear_in(h)
        h = self.dropout(h)
        inter_h = h[inter_mask]
        inter_channel_attr = channel_attr[inter_mask]
        inter_channel_weights = channel_weights[inter_mask]
        
        # Tokenize ellipsoids if they are provided
        ellip_tokens, mu, Sigma = None, None, None
        if ellipsoids is not None:
            ellip_tokens = self.ellipsoid_conditioner.tokenize_ellipsoids(ellipsoids)
            mu = ellipsoids['mu']
            Sigma = ellipsoids['Sigma']

        ctx_states, ctx_coords, inter_coords = [], [], []
        for i in range(0, self.n_layers):
            # This is your IPA step
            h, x = self._modules[f'ctx_gcl_{i}'](
                h, ctx_edges, x, channel_attr, channel_weights,
                edge_attr=ctx_edge_attr,
                H_sv=H_sv, H_se=ctx_H_se)

            # Inject ellipsoid information
            if ellipsoids is not None:
                h = self.ellipsoid_conditioner.inject(h, R, t, ellip_tokens, mu, Sigma)
            
            # Concatenate and transform
            h = self.ellipsoid_conditioner.concat_and_transform(h, ellip_tokens)

            # synchronization of the shadow paratope (native -> shadow)
            inter_h = inter_h.clone()
            inter_h[inter_update_mask] = h[update_mask]
            inter_h, inter_x = self._modules[f'inter_gcl_{i}'](
                inter_h, inter_edges, inter_x, inter_channel_attr, inter_channel_weights,
                H_sv=H_sv, H_se=int_H_se
            )
            # synchronization of the shadow paratope (shadow -> native)
            h = h.clone()
            h[inter_mask] = inter_h
            ctx_states.append(h)
            ctx_coords.append(x)
            inter_coords.append(inter_x)

        # The rest of the forward pass remains the same
        h, x = self.out_layer(
            h, ctx_edges, x, channel_attr, channel_weights,
            edge_attr=ctx_edge_attr, H_sv=H_sv, H_se=ctx_H_se)
        ctx_states.append(h)
        ctx_coords.append(x)
        if self.dense:
            h = torch.cat(ctx_states, dim=-1)
            x = torch.mean(torch.stack(ctx_coords), dim=0)
            inter_x = torch.mean(torch.stack(inter_coords), dim=0)
        h = self.dropout(h)
        h = self.linear_out(h)
        
        # The rigid_update and edge_update are likely handled outside this encoder
        return h, x, inter_x
