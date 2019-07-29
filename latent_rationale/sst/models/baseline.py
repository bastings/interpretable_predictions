#!/usr/bin/env python

import torch
from torch import nn
from latent_rationale.common.util import get_encoder
from latent_rationale.common.classifier import Classifier
from latent_rationale.common.util import get_z_stats


class Baseline(nn.Module):
    """
    Baseline
    Encode sentence x with a (Bi-)LSTM.
    Classify from final state(s).
    """

    def __init__(self, vocab_size, emb_size, hidden_size, output_size,
                 vocab, layer="rcnn", dropout=0.2):
        super(Baseline, self).__init__()

        self.vocab = vocab
        self.hidden_size = hidden_size

        self.embed = embed = nn.Embedding(vocab_size, emb_size, padding_idx=1)

        self.classifier = Classifier(
            embed=embed, hidden_size=hidden_size, output_size=output_size,
            dropout=dropout, layer=layer, nonlinearity="softmax")

        self.alphas = None  # attention scores
        self.criterion = nn.NLLLoss(reduction='none')

    def forward(self, x, **kwargs):
        """
        Generate a sequence of zs with the Generator.
        Then predict with sentence x (zeroed out with z) using Encoder.

        :param x: [B, T] (that is, batch-major is assumed)
        :return:
        """
        mask = (x != 1)  # [B,T]
        y = self.classifier(x, mask)
        return y

    def predict(self, logits):
        """
        Predict.
        :param logits:
        :return:
        """
        return logits.argmax(dim=-1)

    def get_loss(self, logits, targets, **kwargs):

        optional = dict()
        loss = self.criterion(logits, targets)

        return loss.mean(), optional
