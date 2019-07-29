#!/usr/bin/env python

import numpy as np

import torch
from torch import nn
from torch.nn.functional import softplus

from latent_rationale.common.util import get_z_stats
from latent_rationale.common.classifier import Classifier
from latent_rationale.common.generator import IndependentGenerator
from latent_rationale.common.generator import DependentGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RLModel(nn.Module):
    """
    Reimplementation of Lei et al. (2016). Rationalizing Neural Predictions
    for Stanford Sentiment.
    (Does classfication instead of regression.)

    Consists of:
    - Encoder that computes p(y | x, z)
    - Generator that computes p(z | x) independently or dependently with an RNN.
    """

    def __init__(self,
                 vocab:       object = None,
                 vocab_size:  int = 0,
                 emb_size:    int = 200,
                 hidden_size: int = 200,
                 output_size: int = 1,
                 dropout:     float = 0.1,
                 layer:       str = "lstm",
                 dependent_z: bool = False,
                 sparsity:    float = 0.0,
                 coherence:   float = 0.0,
                 ):

        super(RLModel, self).__init__()

        self.vocab = vocab
        self.sparsity = sparsity
        self.coherence = coherence

        self.embed = embed = nn.Embedding(vocab_size, emb_size, padding_idx=1)

        self.encoder = Classifier(
            embed=embed, hidden_size=hidden_size, output_size=output_size,
            dropout=dropout, layer=layer, nonlinearity="softmax")

        if dependent_z:
            self.generator = DependentGenerator(
                embed=embed, hidden_size=hidden_size,
                dropout=dropout, layer=layer)
        else:
            self.generator = IndependentGenerator(
                embed=embed, hidden_size=hidden_size,
                dropout=dropout, layer=layer)

        self.criterion = nn.NLLLoss(reduction='none')

    def lagrange_parameters(self):
        return []

    @property
    def z(self):
        return self.generator.z

    @property
    def z_layer(self):
        return self.generator.z_layer

    def predict(self, logits, **kwargs):
        """
        Predict deterministically.
        :param x:
        :return: predictions, optional (dict with optional statistics)
        """
        assert not self.training, "should be in eval mode for prediction"
        return logits.argmax(-1)

    def forward(self, x, **kwargs):
        """
        Generate a sequence of zs with the Generator.
        Then predict with sentence x (zeroed out with z) using Encoder.

        :param x: [B, T] (that is, batch-major is assumed)
        :return:
        """
        mask = (x != 1)  # [B,T]
        z = self.generator(x, mask)
        y = self.encoder(x, mask, z)
        return y

    def get_loss(self, logits, targets, mask=None, **kwargs):
        """
        This computes the loss for the whole model.
        We stick to the variable names of the original code by Tao Lei
        as much as possible.

        :param logits:
        :param targets:
        :param mask:
        :param kwargs:
        :return:
        """
        assert mask is not None, "provide mask"

        optional = {}
        sparsity = self.sparsity
        coherence = self.coherence

        loss_vec = self.criterion(logits, targets)  # [B]

        # main MSE loss for p(y | x,z)
        loss = loss_vec.mean()        # [1]
        optional["ce"] = loss.item()  # [1]

        # compute generator loss
        z = self.generator.z.squeeze(1).squeeze(-1)  # [B, T]

        if self.training:
            num_0, num_c, num_1, total = get_z_stats(self.generator.z, mask)
            optional["p1"] = num_1 / float(total)

        # get P(z = 0 | x) and P(z = 1 | x)
        if len(self.generator.z_dists) == 1:  # independent z
            m = self.generator.z_dists[0]
            logp_z0 = m.log_prob(0.).squeeze(2)  # [B,T], log P(z = 0 | x)
            logp_z1 = m.log_prob(1.).squeeze(2)  # [B,T], log P(z = 1 | x)
        else:  # for dependent z case, first stack all log probs
            logp_z0 = torch.stack(
                [m.log_prob(0.) for m in self.generator.z_dists], 1).squeeze(2)
            logp_z1 = torch.stack(
                [m.log_prob(1.) for m in self.generator.z_dists], 1).squeeze(2)

        # compute log p(z|x) for each case (z==0 and z==1) and mask
        logpz = torch.where(z == 0, logp_z0, logp_z1)
        logpz = torch.where(mask, logpz, logpz.new_zeros([1]))

        # sparsity regularization
        zsum = z.sum(1)  # [B]
        zdiff = z[:, 1:] - z[:, :-1]
        zdiff = zdiff.abs().sum(1)  # [B]

        zsum_cost = sparsity * zsum.mean(0)
        optional["zsum_cost"] = zsum_cost.item()

        zdiff_cost = coherence * zdiff.mean(0)
        optional["zdiff_cost"] = zdiff_cost.mean().item()

        sparsity_cost = zsum_cost + zdiff_cost
        optional["sparsity_cost"] = sparsity_cost.item()

        cost_vec = loss_vec.detach() + zsum * sparsity + zdiff * coherence
        cost_logpz = (cost_vec * logpz.sum(1)).mean(0)  # cost_vec is neg reward

        obj = cost_vec.mean()  # MSE with regularizers = neg reward
        optional["obj"] = obj.item()

        # generator cost
        optional["cost_g"] = cost_logpz.item()

        # encoder cost
        optional["cost_e"] = loss.item()

        return loss + cost_logpz, optional
