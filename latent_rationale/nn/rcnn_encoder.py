import torch
from torch import nn
from latent_rationale.nn.rcnn import RCNN


class RCNNEncoder(nn.Module):
    """
    This module encodes a sequence into a single vector using an LSTM.
    """

    def __init__(self, in_features, hidden_size, batch_first: bool = True,
                 bidirectional: bool = True):
        super(RCNNEncoder, self).__init__()
        assert batch_first, "only batch_first=True supported"
        self.rcnn = RCNN(in_features, hidden_size, bidirectional=bidirectional)

    def forward(self, x, mask, lengths):
        """

        :param x: sequence of word embeddings, shape [B, T, E]
        :param mask: byte mask that is 0 for invalid positions, shape [B, T]
        :param lengths: the lengths of each input sequence [B]
        :return:
        """
        return self.rcnn(x, mask, lengths)
