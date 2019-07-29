import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMEncoder(nn.Module):
    """
    This module encodes a sequence into a single vector using an LSTM.
    """

    def __init__(self, in_features, hidden_size: int = 200,
                 batch_first: bool = True,
                 bidirectional: bool = True):
        """
        :param in_features:
        :param hidden_size:
        :param batch_first:
        :param bidirectional:
        """
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(in_features, hidden_size, batch_first=batch_first,
                            bidirectional=bidirectional)

    def forward(self, x, mask, lengths):
        """
        Encode sentence x
        :param x: sequence of word embeddings, shape [B, T, E]
        :param mask: byte mask that is 0 for invalid positions, shape [B, T]
        :param lengths: the lengths of each input sequence [B]
        :return:
        """

        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True)
        outputs, (hx, cx) = self.lstm(packed_sequence)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        # classify from concatenation of final states
        if self.lstm.bidirectional:
            final = torch.cat([hx[-2], hx[-1]], dim=-1)
        else:  # classify from final state
            final = hx[-1]

        return outputs, final
