#!/usr/bin/env python

import torch
from torch import nn
from latent_rationale.common.util import get_encoder


class SimpleClassifier(nn.Module):
    """
    LSTM/RCNN Classifier baseline

    Encode sentence x with a (Bi-)LSTM.
    Classify from final state(s).

    This is not the Lei et al. rationale baseline, for that see `rl.py`.
    """

    def __init__(self, vocab_size, emb_size, hidden_size, output_size,
                 vocab, layer="rcnn", dropout=0.2,
                 bidirectional=True,
                 aspect=-1):
        super(SimpleClassifier, self).__init__()

        assert aspect < output_size, "aspect should be < output_dim"

        self.vocab = vocab
        self.hidden_size = hidden_size
        self.aspect = aspect

        self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=1)
        self.embed_dropout = nn.Dropout(p=dropout)

        self.enc_layer = get_encoder(layer, emb_size, hidden_size,
                                     bidirectional=bidirectional)
        enc_dim = 2 * hidden_size if bidirectional else hidden_size

        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(enc_dim, output_size),
            nn.Sigmoid()
        )

        self.alphas = None  # attention scores
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, x):
        """
        :param x: [B, T] (that is, batch-major is assumed)
        :return:
        """
        mask = (x != 1)
        lengths = mask.sum(1)
        emb = self.embed(x)  # B,T,E
        emb = self.embed_dropout(emb)
        _, final = self.enc_layer(emb, mask, lengths)  # encode sentence
        predictions = self.output_layer(final)

        return predictions

    def get_loss(self, prediction, targets, **kwargs):
        loss = self.criterion(prediction, targets)
        loss = loss.mean()
        optional = dict(mse=loss.item())
        return loss, optional
