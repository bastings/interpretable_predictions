#!/usr/bin/env python

import numpy as np

import torch
from torch import nn

from latent_rationale.common.util import get_z_stats
from latent_rationale.common.classifier import Classifier
from latent_rationale.common.generator import IndependentGenerator
from latent_rationale.common.generator import DependentGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RLModel(nn.Module):
    """
    Reimplementation of Lei et al. (2016). Rationalizing Neural Predictions.

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
                 layer:       str = "rcnn",
                 dependent_z: bool = False,
                 sparsity:    float = 0.0003,
                 coherence:   float = 2.,
                 ):

        super(RLModel, self).__init__()

        self.vocab = vocab
        self.embed = embed = nn.Embedding(vocab_size, emb_size, padding_idx=1)
        self.sparsity = sparsity
        self.coherence = coherence

        self.encoder = Classifier(
            embed=embed, hidden_size=hidden_size, output_size=output_size,
            dropout=dropout, layer=layer)

        if dependent_z:
            self.generator = DependentGenerator(
                embed=embed, hidden_size=hidden_size,
                dropout=dropout, layer=layer)
        else:
            self.generator = IndependentGenerator(
                embed=embed, hidden_size=hidden_size,
                dropout=dropout, layer=layer)

        self.criterion = nn.MSELoss(reduction='none')

    @property
    def z(self):
        return self.generator.z

    @property
    def z_layer(self):
        return self.generator.z_layer

    def predict(self, x, **kwargs):
        """
        Predict deterministically.
        :param x:
        :return: predictions, optional (dict with optional statistics)
        """
        assert not self.training, "should be in eval mode for prediction"

        with torch.no_grad():
            mask = (x != 1)
            predictions = self.forward(x)
            num_0, num_c, num_1, total = get_z_stats(self.z, mask)
            selected = num_1 / float(total)
            optional = dict(selected=selected)
            return predictions, optional

    def forward(self, x):
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

    def get_loss(self, preds, targets, mask=None):
        """
        This computes the loss for the whole model.
        We stick to the variable names of the original code as much as
        possible.

        :param preds:
        :param targets:
        :param mask:
        :param iter_i:
        :return:
        """

        optional = {}
        sparsity = self.sparsity
        coherence = self.coherence

        loss_mat = self.criterion(preds, targets)  # [B, T]

        # main MSE loss for p(y | x,z)
        loss_vec = loss_mat.mean(1)    # [B]
        loss = loss_vec.mean()         # [1]
        optional["mse"] = loss.item()  # [1]

        # coherency is 2*sparsity (after modifying sparsity rate)
        coherent_factor = sparsity * coherence
        
        # compute generator loss
        z = self.generator.z.squeeze()  # [B, T]

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

        zdiff_cost = coherent_factor * zdiff.mean(0)
        optional["zdiff_cost"] = zdiff_cost.mean().item()

        sparsity_cost = zsum_cost + zdiff_cost
        optional["sparsity_cost"] = sparsity_cost.item()

        cost_vec = loss_vec.detach() + zsum * sparsity + zdiff * coherent_factor
        cost_logpz = (cost_vec * logpz.sum(1)).mean(0)  # cost_vec is neg reward

        obj = cost_vec.mean()  # MSE with regularizers = neg reward
        optional["obj"] = obj.item()

        # pred diff doesn't do anything if only 1 aspect being trained
        pred_diff = (preds.max(dim=1)[0] - preds.min(dim=1)[0])
        pred_diff = pred_diff.mean()
        optional["pred_diff"] = pred_diff.item()

        # generator cost
        cost_g = cost_logpz
        optional["cost_g"] = cost_g.item()

        # encoder cost
        cost_e = loss
        optional["cost_e"] = cost_e.item()

        num_0, num_c, num_1, total = get_z_stats(self.generator.z, mask)
        optional["p0"] = num_0 / float(total)
        optional["p1"] = num_1 / float(total)
        optional["selected"] = optional["p1"]

        main_loss = cost_e + cost_g
        return main_loss, optional
