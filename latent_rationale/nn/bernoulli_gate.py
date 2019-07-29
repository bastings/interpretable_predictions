# coding: utf-8

import torch.nn as nn
from torch.nn import Linear, Sequential
from torch.distributions.bernoulli import Bernoulli


class BernoulliGate(nn.Module):
    """
    Computes a Bernoulli Gate
    Assigns a 0 or a 1 to each input word.
    """

    def __init__(self, in_features, out_features=1):
        super(BernoulliGate, self).__init__()

        self.layer = Sequential(
            Linear(in_features, out_features, bias=True)
        )

    def forward(self, x):
        """
        Compute Binomial gate
        :param x: word represenatations [B, T, D]
        :return: gate distribution
        """
        logits = self.layer(x)  # [B, T, 1]
        dist = Bernoulli(logits=logits)
        return dist
