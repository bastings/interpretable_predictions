# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMechanism(nn.Module):

    def __init__(self):
        super(AttentionMechanism, self).__init__()

    def forward(self, *input):
        raise NotImplementedError("Implement this.")


class DotAttention(AttentionMechanism):

    def __init__(self):
        super(DotAttention, self).__init__()

    def forward(self, q, k):
        return q @ k.transpose(1, 2)


class DeepDotAttention(AttentionMechanism):

    def __init__(self, in_features, out_features, dropout=0.2):
        super(DeepDotAttention, self).__init__()

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        self.attention_layer = nn.Sequential(
            nn.Linear(in_features, out_features), self.activation, self.dropout,
            nn.Linear(out_features, out_features), self.activation, self.dropout
        )

    def forward(self, q, k):
        q = self.attention_layer(q)
        k = self.attention_layer(k)
        return q @ k.transpose(1, 2)


class SimpleAttention(AttentionMechanism):
    """
    Implements Simple Independent attention
    """

    def __init__(self, hidden_size=1, key_size=1, dropout=0.2):
        """
        Creates attention mechanism.
        :param hidden_size:
        :param key_size:
        :param dropout:
        """

        super(SimpleAttention, self).__init__()

        self.score_layer = nn.Sequential(
            nn.Linear(key_size, hidden_size, bias=False),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self,
                keys: torch.Tensor = None,
                mask: torch.Tensor = None,
                values: torch.Tensor = None):
        """
        Simple independent attention.

        :param keys: the keys/memory (e.g. encoder states)
            shape: [B, T, Dk]
        :param mask: mask to mask out keys position
            shape: [B, 1, T]
        :param values: values (e.g. typically also encoder states)
            shape: [B, T, D]
        :return: context vector, attention probabilities
        """

        assert mask is not None, "mask is required"

        if values is None:
            values = keys

        # Calculate scores.
        scores = self.score_layer(keys)  # [B,T,1]
        scores = scores.squeeze(2).unsqueeze(1)  # [B,1,T]

        # mask out invalid positions by filling the masked out parts with -inf
        scores = torch.where(mask, scores, scores.new_full([1], float('-inf')))

        # turn scores to probabilities
        alphas = F.softmax(scores, dim=-1)  # batch x 1 x time

        # the context vector is the weighted sum of the values
        context = alphas @ values  # batch x 1 x value_size

        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas

    def __repr__(self):
        return self.__class__.__name__ + "\n" + str(self.score_layer)


class BahdanauAttention(AttentionMechanism):
    """
    Implements Bahdanau (MLP) attention
    """

    def __init__(self, hidden_size=1, key_size=1, query_size=1):
        """
        Creates attention mechanism.
        :param hidden_size:
        :param key_size:
        :param query_size:
        """

        super(BahdanauAttention, self).__init__()

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

        self.proj_keys = None   # to store projected keys
        self.proj_query = None  # projected query

    def forward(self, query: torch.Tensor = None,
                keys: torch.Tensor = None,
                mask: torch.Tensor = None,
                values: torch.Tensor = None):
        """
        Bahdanau additive attention forward pass.

        :param query: the item to compare with the keys/memory
            (e.g. decoder state): shape: [B, 1, Dq]
        :param keys: the keys/memory (e.g. encoder states)
            shape: [B, T, Dk]
        :param mask: mask to mask out keys position
            shape: [B, 1, T]
        :param values: values (e.g. typically also encoder states)
            shape: [B, T, D]
        :return: context vector, attention probabilities
        """

        assert mask is not None, "mask is required"
        assert self.proj_keys is not None,\
            "projection keys have to get pre-computed"

        if values is None:
            values = keys

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        self.compute_proj_query(query)

        # Calculate scores.
        # proj_keys: batch x src_len x hidden_size
        # proj_query: batch x 1 x hidden_size
        scores = self.energy_layer(torch.tanh(self.proj_query + self.proj_keys))
        # scores: batch x src_len x 1

        scores = scores.squeeze(2).unsqueeze(1)
        # scores: batch x 1 x time

        # mask out invalid positions by filling the masked out parts with -inf
        scores = torch.where(mask, scores, scores.new_full([1], float('-inf')))

        # turn scores to probabilities
        alphas = F.softmax(scores, dim=-1)  # batch x 1 x time

        # the context vector is the weighted sum of the values
        context = alphas @ values  # batch x 1 x value_size

        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas

    def compute_proj_keys(self, keys):
        """
        Compute the projection of the keys.
        Is efficient if pre-computed before receiving individual queries.
        :param keys:
        :return:
        """
        self.proj_keys = self.key_layer(keys)

    def compute_proj_query(self, query):
        """
        Compute the projection of the query.
        :param query:
        :return:
        """
        self.proj_query = self.query_layer(query)

    def __repr__(self):
        return self.__class__.__name__
