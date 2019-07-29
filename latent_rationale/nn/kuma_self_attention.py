# coding: utf-8
import torch
import torch.nn as nn
from torch.nn.functional import softplus
from torch.nn import Linear, Sequential, Tanh, ReLU, Dropout, Embedding

from latent_rationale.nn.position import get_relative_positions
from latent_rationale.nn.kuma import Kuma, HardKuma


class KumaSelfAttention(nn.Module):
    """
    Computes Hard Kumaraswamy Attention
    """

    def __init__(self, in_features, out_features, support=(-0.1, 1.1),
                 dropout=0.2, dist_type="hardkuma", add_rel_dist=True,
                 max_relative_distance=11, mask_diag=False, dist_embed=None):
        super(KumaSelfAttention, self).__init__()

        self.dist_type = dist_type
        self.activation = ReLU()
        self.dropout = Dropout(p=dropout)

        self.max_relative_distance = max_relative_distance
        self.mask_diag = mask_diag  # mask diagonal
        self.dist_embed = dist_embed
        self.add_rel_dist = add_rel_dist

        self.attention_layer_a = Sequential(
            Linear(in_features, out_features), self.activation, self.dropout,
            Linear(out_features, out_features), self.activation, self.dropout
        )
        self.attention_layer_b = Sequential(
            Linear(in_features, out_features), self.activation, self.dropout,
            Linear(out_features, out_features), self.activation, self.dropout
        )
        self.support = support

        self.dist = None

    def _mask_diagonal(self, x, mask_value=0.):
        """block the diagonal so a word does not self-align"""
        eye = torch.eye(x.size(1), dtype=torch.uint8, device=x.device)
        return torch.where(eye, x.new_full([1], mask_value), x)

    def _add_rel_dists(self, x):
        """add matrix of relative distances"""
        rel_dists = get_relative_positions(
            x.size(1), self.max_relative_distance, device=x.device)
        rel_dists = self.dist_embed(rel_dists).squeeze(-1).unsqueeze(0)
        return x + rel_dists

    def forward(self, q, k):

        q_a = self.attention_layer_a(q)
        k_a = self.attention_layer_a(k)

        q_b = self.attention_layer_b(q)
        k_b = self.attention_layer_b(k)

        a = q_a @ k_a.transpose(1, 2)
        b = q_b @ k_b.transpose(1, 2)

        # add relative distances
        if self.add_rel_dist:
            a = self._add_rel_dists(a)
            b = self._add_rel_dists(b)

        a = softplus(a)
        b = softplus(b)

        a = a.clamp(0.01, 100.)
        b = b.clamp(0.01, 100.)

        # we return a distribution (from which we can sample if we want)
        if self.dist_type == "kuma":
            dist = Kuma([a, b])
        elif self.dist_type == "hardkuma":
            dist = HardKuma([a, b], support=self.support)
        else:
            raise ValueError("unknown dist")

        self.dist = dist

        if self.training:  # sample
            att = dist.sample()
        else:  # predict deterministically
            p0 = dist.pdf(q.new_zeros(()))
            p1 = dist.pdf(q.new_ones(()))
            pc = 1. - p0 - p1  # prob. of sampling a continuous value [B, M]
            zero_one = torch.where(
                p0 > p1, q.new_zeros([1]), q.new_ones([1]))
            att = torch.where(pc < 0.5, zero_one, dist.mean())  # [B, M]

        if self.mask_diag:
            att = self._mask_diagonal(att, mask_value=0.)

        return att  # [B, M]
