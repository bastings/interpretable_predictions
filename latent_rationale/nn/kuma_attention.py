# coding: utf-8
import torch
import torch.nn as nn
from torch.nn.functional import softplus
from torch.nn import Linear, Sequential, ReLU, Dropout

from latent_rationale.nn.kuma import Kuma, HardKuma

MIN_CLAMP = 1e-3
MAX_CLAMP = 100.


class KumaAttention(nn.Module):
    """
    Computes Hard Kumaraswamy Attention
    """

    def __init__(self, in_features, out_features, support=(-0.1, 1.1),
                 dropout=0.2, dist_type="hardkuma"):
        super(KumaAttention, self).__init__()

        self.dist_type = dist_type
        self.activation = ReLU()
        self.dropout = Dropout(p=dropout)

        self.a_layer = Sequential(
            Linear(in_features, out_features), self.activation, self.dropout,
            Linear(out_features, out_features), self.activation, self.dropout
        )
        self.b_layer = Sequential(
            Linear(in_features, out_features), self.activation, self.dropout,
            Linear(out_features, out_features), self.activation, self.dropout
        )
        self.support = support

        self.dist = None

    def forward(self, q, k):
        q_a = self.a_layer(q)
        k_a = self.a_layer(k)

        q_b = self.b_layer(q)
        k_b = self.b_layer(k)

        a = q_a @ k_a.transpose(1, 2)
        b = q_b @ k_b.transpose(1, 2)

        a = softplus(a)
        b = softplus(b)

        a = a.clamp(MIN_CLAMP, MAX_CLAMP)
        b = b.clamp(MIN_CLAMP, MAX_CLAMP)

        # we return a distribution (from which we can sample if we want)
        if self.dist_type == "kuma":
            dist = Kuma([a, b])
        elif self.dist_type == "hardkuma":
            dist = HardKuma([a, b], support=self.support)
        else:
            raise ValueError("unknown dist")

        self.dist = dist

        if self.training:  # sample
            return dist.sample()
        else:  # predict deterministically
            p0 = dist.pdf(q.new_zeros(()))
            p1 = dist.pdf(q.new_ones(()))
            pc = 1. - p0 - p1  # prob. of sampling a continuous value [B, M]
            zero_one = torch.where(
                p0 > p1, q.new_zeros([1]), q.new_ones([1]))
            z = torch.where((pc > p0) & (pc > p1),
                            dist.mean(), zero_one)  # [B, M]
            return z  # [B, M]
