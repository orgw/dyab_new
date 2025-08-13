#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

import numpy as np
from data import VOCAB
from utils.nn_utils import SeparatedAminoAcidFeature, ProteinFeature
from utils.nn_utils import GMEdgeConstructor, SeperatedCoordNormalizer
from utils.nn_utils import _knn_edges, diff_l1_loss
from evaluation.rmsd import kabsch_torch

from ..modules.dyAb_encoder import dyAbEncoder
from ..modules.dyAb_egnn import dyAbEGNN
from ..modules.pie import PIE



class GaussianSmearing(torch.nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1) + 1E-6
        return torch.exp(self.coeff * torch.pow(dist, 2))



class dyAbModel(nn.Module):
    def __init__(self, embed_size, hidden_size, n_channel, num_classes,
                 mask_id=VOCAB.get_mask_idx(), k_neighbors=9, bind_dist_cutoff=6,
                 n_layers=3, iter_round=3, dropout=0.1, struct_only=False,
                 backbone_only=False, fix_channel_weights=False, pred_edge_dist=True,
                 cdr_type='H3', paratope='H3', relative_position=False, flexible=False, module_type=0) -> None:
        super().__init__()
        self.mask_id = mask_id
        self.num_classes = num_classes
        self.bind_dist_cutoff = bind_dist_cutoff
        self.k_neighbors = k_neighbors
        self.round = iter_round
        self.struct_only = struct_only

        # options
        self.backbone_only = backbone_only
        self.fix_channel_weights = fix_channel_weights

        self.pred_edge_dist = pred_edge_dist
        
        if self.backbone_only:
            n_channel = 4
        self.cdr_type = cdr_type
        self.paratope = paratope
        self.flexible = flexible

        atom_embed_size = embed_size // 4
        self.aa_feature = SeparatedAminoAcidFeature(
            embed_size, atom_embed_size,
            relative_position=relative_position,
            edge_constructor=GMEdgeConstructor,
            fix_atom_weights=fix_channel_weights,
            backbone_only=backbone_only
        )
        self.protein_feature = ProteinFeature(backbone_only=backbone_only)
        if self.pred_edge_dist:  # use predicted dist for KNN-graph at the interface
            self.edge_dist_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(2 * hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )
            # this GNN encodes the initial hidden states for initial edge distance prediction
            self.init_gnn = dyAbEGNN(
                embed_size, hidden_size, hidden_size, n_channel,
                channel_nf=atom_embed_size, radial_nf=hidden_size,
                in_edge_nf=0, n_layers=n_layers, residual=True,
                dropout=dropout, dense=False, init=True)
        if not struct_only:
            self.ffn_residue = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, self.num_classes)
            )

            self.ffn_residue2 = nn.Sequential(
                nn.SiLU(),
                nn.Linear(embed_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, self.num_classes)
            )

            self.ffn_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, embed_size),
            )

            self.ffn_pls = nn.Sequential(
                nn.SiLU(),
                nn.Linear(embed_size, hidden_size),
            )
        else:
            self.prmsd_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )
            self.prmsd_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )
        self.gnn = dyAbEncoder(
            embed_size, hidden_size, hidden_size, n_channel,
            channel_nf=atom_embed_size, radial_nf=hidden_size,
            in_edge_nf=0, n_layers=n_layers, residual=True,
            dropout=dropout, dense=False, init=False)
        
        self.normalizer = SeperatedCoordNormalizer()

        self.pie = PIE(hidden_size, hidden_size)
        time_mapping = GaussianSmearing(0.0, 1, embed_size)
        self.time_emb = nn.Sequential(
            time_mapping,
            nn.Linear(embed_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_size)
        )

        self.module_type = module_type

        # training related cache
        self.batch_constants = {}

    def init_mask(self, X, S, cmask, smask, template):
        if not self.struct_only:
            S[smask] = self.mask_id
        X[cmask] = template
        return X, S

    def message_passing(self, X, S, residue_pos, interface_X, paratope_mask, batch_id, t, smooth_prob=None, smooth_mask=None):
        # embeddings
        H_0, (ctx_edges, inter_edges), (atom_embeddings, atom_weights) = self.aa_feature(X, S, batch_id, self.k_neighbors, residue_pos, smooth_prob=smooth_prob, smooth_mask=smooth_mask)
        time_emb = self.time_emb(t)

        if self.pred_edge_dist:
            # replace the MLP with gnn for initial edge distance prediction
            edge_H, dumb_X = self.init_gnn(H_0, X, ctx_edges,
                                    channel_attr=atom_embeddings,
                                    channel_weights=atom_weights)
            X = X + dumb_X * 0  # to cheat the autograd check

        # update coordination of the global node
        X = self.aa_feature.update_globel_coordinates(X, S)

        # prepare local complex
        # local_mask is the paratope
        local_mask = self.batch_constants['local_mask']
        local_is_ab = self.batch_constants['local_is_ab']
        local_batch_id = self.batch_constants['local_batch_id']
        local_X = X[local_mask].clone()
        # prepare local complex edges
        local_ctx_edges = self.batch_constants['local_ctx_edges']  # [2, Ec]
        local_inter_edges = self.batch_constants['local_inter_edges']  # [2, Ei]
        atom_pos = self.aa_feature._construct_atom_pos(S[local_mask])
        offsets, max_n, gni2lni = self.batch_constants['local_edge_infos']
        # for context edges, use edges in the native paratope
        local_ctx_edges = _knn_edges(
            local_X, atom_pos, local_ctx_edges.T,
            self.aa_feature.atom_pos_pad_idx, self.k_neighbors,
            (offsets, local_batch_id, max_n, gni2lni))
        # for interative edges, use edges derived from the predicted distance
        # using the paratope as the epitope in the antibody
        local_X[local_is_ab] = interface_X
        if self.pred_edge_dist:
            local_H = edge_H[local_mask]
            src_H, dst_H = local_H[local_inter_edges[0]], local_H[local_inter_edges[1]]
            p_edge_dist = self.edge_dist_ffn(torch.cat([src_H, dst_H], dim=-1)) +\
                          self.edge_dist_ffn(torch.cat([dst_H, src_H], dim=-1))  # perm-invariant
            p_edge_dist = p_edge_dist.squeeze()
        else:
            p_edge_dist = None
        local_inter_edges = _knn_edges(
            local_X, atom_pos, local_inter_edges.T,
            self.aa_feature.atom_pos_pad_idx, self.k_neighbors,
            (offsets, local_batch_id, max_n, gni2lni), given_dist=p_edge_dist)
            
        local_edges = torch.cat([local_ctx_edges, local_inter_edges], dim=1)

        H_sv, int_H_se, ctx_H_se = self.pie(X, local_edges, ctx_edges)

        pre_H_0 = H_0.clone()
        H_0 = H_0 + time_emb

        # message passing
        # the model predicts the seq emb, the complete ag, the paratope
        H, pred_X, pred_local_X = self.gnn(H_0, X, ctx_edges,
                                           local_mask, local_X, local_edges,
                                           paratope_mask, local_is_ab,
                                           channel_attr=atom_embeddings,
                                           channel_weights=atom_weights,
                                           H_sv=H_sv, ctx_H_se=ctx_H_se, int_H_se=int_H_se)
        # using the predicted paratope
        interface_X = pred_local_X[local_is_ab]
        # pred_logits = self.ffn_residue2(pre_H_0 * torch.sigmoid(self.ffn_proj(H)))
        pred_logits = None if self.struct_only else self.ffn_residue2(pre_H_0 * torch.sigmoid(self.ffn_proj(H)))


        return pred_logits, pred_X, interface_X, H, p_edge_dist  # [N, num_classes], [N, n_channel, 3], [Ncdr, n_channel, 3], [N, hidden_size]
    
    @torch.no_grad()
    def init_interface(self, X, S, paratope_mask, batch_id, init_noise=None):
        # interface is initialized around the paratope
        ag_centers = X[S == self.aa_feature.boa_idx][:, 0]  # [bs, 3]
        init_local_X = torch.zeros_like(X[paratope_mask]) + ag_centers[batch_id[paratope_mask]].unsqueeze(1)
        noise = torch.randn_like(init_local_X) if init_noise is None else init_noise
        ca_noise = noise[:, 1]
        noise = noise / 10  + ca_noise.unsqueeze(1) # scale other atoms
        noise[:, 1] = ca_noise
        init_local_X += noise
        return init_local_X

    @torch.no_grad()
    def _prepare_batch_constants(self, S, paratope_mask, lengths):
        # generate batch id
        batch_id = torch.zeros_like(S)  # [N]
        batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_id.cumsum_(dim=0)  # [N], item idx in the batch
        self.batch_constants['batch_id'] = batch_id
        self.batch_constants['batch_size'] = torch.max(batch_id) + 1

        segment_ids = self.aa_feature._construct_segment_ids(S)
        self.batch_constants['segment_ids'] = segment_ids

        # interface relatd
        is_ag = segment_ids == self.aa_feature.ag_seg_id
        not_ag_global = S != self.aa_feature.boa_idx
        local_mask = torch.logical_or(
            paratope_mask, torch.logical_and(is_ag, not_ag_global)
        )
        local_segment_ids = segment_ids[local_mask]
        local_is_ab = local_segment_ids != self.aa_feature.ag_seg_id
        local_batch_id = batch_id[local_mask]
        self.batch_constants['is_ag'] = is_ag
        self.batch_constants['local_mask'] = local_mask
        self.batch_constants['local_is_ab'] = local_is_ab
        self.batch_constants['local_batch_id'] = local_batch_id
        self.batch_constants['local_segment_ids'] = local_segment_ids
        # interface local edges
        (row, col), (offsets, max_n, gni2lni) = self.aa_feature.edge_constructor.get_batch_edges(local_batch_id)
        row_segment_ids, col_segment_ids = local_segment_ids[row], local_segment_ids[col]
        is_ctx = row_segment_ids == col_segment_ids
        is_inter = torch.logical_not(is_ctx)

        self.batch_constants['local_ctx_edges'] = torch.stack([row[is_ctx], col[is_ctx]])  # [2, Ec]
        self.batch_constants['local_inter_edges'] = torch.stack([row[is_inter], col[is_inter]])  # [2, Ei]
        self.batch_constants['local_edge_infos'] = (offsets, max_n, gni2lni)

        interface_batch_id = batch_id[paratope_mask]
        self.batch_constants['interface_batch_id'] = interface_batch_id
    
    def _clean_batch_constants(self):
        self.batch_constants = {}

    @torch.no_grad()
    def _get_inter_edge_dist(self, X, S):
        local_mask = self.batch_constants['local_mask']
        atom_pos = self.aa_feature._construct_atom_pos(S[local_mask])
        src_dst = self.batch_constants['local_inter_edges'].T
        dist = X[local_mask][src_dst]  # [Ef, 2, n_channel, 3]
        dist = dist[:, 0].unsqueeze(2) - dist[:, 1].unsqueeze(1)  # [Ef, n_channel, n_channel, 3]
        dist = torch.norm(dist, dim=-1)  # [Ef, n_channel, n_channel]
        pos_pad = atom_pos[src_dst] == self.aa_feature.atom_pos_pad_idx # [Ef, 2, n_channel]
        pos_pad = torch.logical_or(pos_pad[:, 0].unsqueeze(2), pos_pad[:, 1].unsqueeze(1))  # [Ef, n_channel, n_channel]
        dist = dist + pos_pad * 1e10  # [Ef, n_channel, n_channel]
        dist = torch.min(dist.reshape(dist.shape[0], -1), dim=1)[0]  # [Ef]
        return dist
    
    def _forward(self, X, S, cmask, smask, paratope_mask, residue_pos, template, lengths, init_noise=None, sample=False):
        batch_id = self.batch_constants['batch_id']

        # mask sequence and initialize coordinates with template
        true_X, true_S = X.clone(), S.clone()
        try:
            X, S = self.init_mask(X, S, cmask, smask, template)
        except:
            print(f'Error in init_mask: {X.shape}, {S.shape}, {cmask.shape}, {smask.shape}, {template.shape}')
            import pdb; pdb.set_trace()
        # normalize
        X = self.normalizer.centering(X, S, batch_id, self.aa_feature)
        X = self.normalizer.normalize(X)
        # update center
        X = self.aa_feature.update_globel_coordinates(X, S)

        # prepare initial interface
        interface_X = self.init_interface(X, S, paratope_mask, batch_id, init_noise)

        if sample is False:
            t = torch.rand(self.batch_constants['batch_id'].max() + 1, device=X.device)
            t = torch.gather(t, 0, self.batch_constants['batch_id'])

            true_X = self.normalizer.centering(true_X, true_S, batch_id, self.aa_feature)
            true_X = self.normalizer.normalize(true_X)
            true_X = self.aa_feature.update_globel_coordinates(true_X, true_S)
            pseudo_X = t[:, None, None] * true_X + (1 - t[:, None, None]) * X
    
            pseudo_S = torch.zeros((S.shape[0], self.num_classes), device=S.device) + 1. / self.num_classes
            pseudo_S[smask, :] = t[smask, None] * F.one_hot(true_S[smask], num_classes=self.num_classes) + (1 - t[smask, None]) * pseudo_S[smask]

            pred_S_logits, pred_X, interface_X, H, edge_dist = self.message_passing(pseudo_X, S, residue_pos, interface_X, paratope_mask, batch_id, t, pseudo_S[smask], smask)
            if not self.struct_only:
                S = S.clone()
                S[smask] = torch.argmax(pred_S_logits[smask], dim=-1)
            r_pred_S_logits = [(pred_S_logits, smask)]
            r_interface_X = [interface_X]
            r_edge_dist = [edge_dist]
        else:
            r_pred_S_logits = []
            r_interface_X = [interface_X.clone()]  # init
            r_edge_dist = []
            steps = 11
            init_X = X.clone()

            S_prob = torch.zeros((S.shape[0], self.num_classes), device=S.device) + 1. / self.num_classes
            init_S_prob = S_prob.clone()

            for t in np.linspace(0, 1, steps)[:-2]:
                tt = torch.tensor([t], dtype=torch.float32, device=X.device).repeat(X.shape[0], 1, 1)
                pred_S_logits, pred_X, interface_X, H, edge_dist = self.message_passing(X, S, residue_pos, interface_X, paratope_mask, batch_id, tt, S_prob[smask], smask)

                r_interface_X.append(interface_X.clone())
                r_pred_S_logits.append((pred_S_logits, smask))
                r_edge_dist.append(edge_dist)
                # 1. update X
                X = X.clone()
                delta_t = 1. / (steps - 1)
                X[cmask] = X[cmask] + delta_t * (pred_X[cmask] - init_X[cmask])
                X = self.aa_feature.update_globel_coordinates(X, S)
                # 2. update S prob
                # S_prob[smask] = S_prob[smask] + delta_t * (torch.softmax(pred_S_logits[smask], dim=-1) - init_S_prob[smask])

                if not self.struct_only:
                    S_prob[smask] = S_prob[smask] + delta_t * (torch.softmax(pred_S_logits[smask], dim=-1) - init_S_prob[smask])
                    # 2. update S
                    S = S.clone()
                    if t == np.linspace(0, 1, steps)[-3]:
                        S[smask] = torch.argmax(pred_S_logits[smask], dim=-1)

        interface_batch_id = self.batch_constants['interface_batch_id']

        if self.struct_only:
            # predicted rmsd
            prmsd = self.prmsd_ffn(H[cmask]).squeeze()  # [N_ab]
        else:
            prmsd = None

        # uncentering and unnormalize
        pred_X = self.normalizer.unnormalize(pred_X)
        pred_X = self.normalizer.uncentering(pred_X, batch_id)
        for i, interface_X in enumerate(r_interface_X):
            interface_X = self.normalizer.unnormalize(interface_X)
            interface_X = self.normalizer.uncentering(interface_X, interface_batch_id, _type=4)
            r_interface_X[i] = interface_X
        self.normalizer.clear_cache()

        return H, S, r_pred_S_logits, pred_X, r_interface_X, r_edge_dist, prmsd

    def forward(self, X, S, cmask, smask, paratope_mask, residue_pos, template, lengths, xloss_mask, context_ratio=0):
        '''
        :param X: [N, n_channel, 3], Cartesian coordinates
        :param context_ratio: float, rate of context provided in masked sequence, should be [0, 1) and anneal to 0 in training
        '''
        if self.backbone_only:
            X, template = X[:, :4], template[:, :4]  # backbone
            xloss_mask = xloss_mask[:, :4]
        # clone ground truth coordinates, sequence
        true_X, true_S = X.clone(), S.clone()

        # prepare constants
        self._prepare_batch_constants(S, paratope_mask, lengths)
        batch_id = self.batch_constants['batch_id']

        # provide some ground truth for annealing sequence training
        if context_ratio > 0:
            not_ctx_mask = torch.rand_like(smask, dtype=torch.float) >= context_ratio
            smask = torch.logical_and(smask, not_ctx_mask)

        # get results
        _, pred_S, r_pred_S_logits, pred_X, r_interface_X, r_edge_dist, prmsd = self._forward(X, S, cmask, smask, paratope_mask, residue_pos, template, lengths)

        # sequence negtive log likelihood
        snll, total = 0, 0
        if not self.struct_only:
            for logits, mask in r_pred_S_logits:
                snll = snll + F.cross_entropy(logits[mask], true_S[mask], reduction='sum')
                total = total + mask.sum()
            snll = snll / total

        # structure loss
        struct_loss, struct_loss_details, bb_rmsd, ops = self.protein_feature.structure_loss(pred_X, true_X, true_S, cmask, batch_id, xloss_mask, self.aa_feature)

        # docking loss
        gt_interface_X = true_X[paratope_mask]
        # 1. interface loss (shadow paratope)
        interface_atom_pos = self.aa_feature._construct_atom_pos(true_S[paratope_mask])
        interface_atom_mask = interface_atom_pos != self.aa_feature.atom_pos_pad_idx
        interface_loss = diff_l1_loss(
            r_interface_X[-1][interface_atom_mask],
            gt_interface_X[interface_atom_mask])
        # 2. edge dist loss
        if self.pred_edge_dist:
            gt_edge_dist = self._get_inter_edge_dist(self.normalizer.normalize(true_X), true_S)
            ed_loss, r_ed_losses = 0, []
            for edge_dist in r_edge_dist:
                r_ed_loss = diff_l1_loss(edge_dist, gt_edge_dist)
                ed_loss = ed_loss + r_ed_loss
                r_ed_losses.append(r_ed_loss)
        else:
            r_ed_losses = [0 for _ in range(self.round)]
            ed_loss = 0
        dock_loss = interface_loss + ed_loss

        if self.struct_only:
            # predicted rmsd
            prmsd_loss = diff_l1_loss(prmsd, bb_rmsd)
            pdev_loss = prmsd_loss
        else:
            pdev_loss, prmsd_loss = None, None

        # comprehensive loss
        loss = snll + struct_loss + dock_loss + (0 if pdev_loss is None else pdev_loss)

        self._clean_batch_constants()

        # AAR
        with torch.no_grad():
            aa_hit = pred_S[smask] == true_S[smask]
            aar = aa_hit.long().sum() / aa_hit.shape[0]

        return loss, (snll, aar), (struct_loss, *struct_loss_details), (dock_loss, interface_loss, ed_loss, r_ed_losses), (pdev_loss, prmsd_loss)

    def sample(self, X, S, cmask, smask, paratope_mask, residue_pos, template, lengths, init_noise=None, return_hidden=False):
        if self.backbone_only:
            X, template = X[:, :4], template[:, :4]  # backbone
        gen_X, gen_S = X.clone(), S.clone()
        
        # prepare constants
        self._prepare_batch_constants(S, paratope_mask, lengths)

        batch_id = self.batch_constants['batch_id']
        batch_size = self.batch_constants['batch_size']
        segment_ids = self.batch_constants['segment_ids']
        interface_batch_id = self.batch_constants['interface_batch_id']
        is_ab = segment_ids != self.aa_feature.ag_seg_id
        s_batch_id = batch_id[smask]

        best_metric = torch.ones(batch_size, dtype=torch.float, device=X.device) * 1e10
        interface_cmask = paratope_mask[cmask]

        n_tries = 10 if self.struct_only else 1
        for i in range(n_tries):
        
            # generate
            H, pred_S, r_pred_S_logits, pred_X, r_interface_X, _, prmsd = self._forward(X, S, cmask, smask, paratope_mask, residue_pos, template, lengths, init_noise, sample=True)

            # PPL or PRMSD
            if not self.struct_only:
                S_logits = r_pred_S_logits[-1][0][smask]
                if self.module_type == 1:
                    S_logits[torch.isnan(S_logits)] = 0.
                S_probs = torch.max(torch.softmax(S_logits, dim=-1), dim=-1)[0]
                nlls = -torch.log(S_probs)
                metric = scatter_mean(nlls, s_batch_id)  # [batch_size]
            else:
                metric = scatter_mean(prmsd[interface_cmask], interface_batch_id)  # [batch_size]

            update = metric < best_metric
            cupdate = cmask & update[batch_id]
            supdate = smask & update[batch_id]
            # update metric history
            best_metric[update] = metric[update]

            # 1. set generated part
            gen_X[cupdate] = pred_X[cupdate]
            if not self.struct_only:
                gen_S[supdate] = pred_S[supdate]
        
            interface_X = r_interface_X[-1]
            # 2. align by cdr
            for i in range(batch_size):
                if not update[i]:
                    continue
                # 1. align CDRH3
                is_cur_graph = batch_id == i
                cdrh3_cur_graph = torch.logical_and(is_cur_graph, paratope_mask)
                # select the predicted cdrh3 region
                ori_cdr = gen_X[cdrh3_cur_graph][:, :4]  # backbone
                # align with the paratope interface
                pred_cdr = interface_X[interface_batch_id == i][:, :4]
                _, R, t = kabsch_torch(ori_cdr.reshape(-1, 3), pred_cdr.reshape(-1, 3))

                # 2. tranform antibody
                is_cur_ab = is_cur_graph & is_ab
                ab_X = torch.matmul(gen_X[is_cur_ab], R.T) + t
                gen_X[is_cur_ab] = ab_X

        self._clean_batch_constants()

        if return_hidden:
            return gen_X, gen_S, metric, H
        return gen_X, gen_S, metric