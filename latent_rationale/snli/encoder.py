import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RecurrentEncoder(nn.Module):

    def __init__(self, cfg):
        super(RecurrentEncoder, self).__init__()
        self.cfg = cfg
        input_size = cfg.proj_size if cfg.projection else cfg.embed_size
        dropout = 0 if cfg.n_layers == 1 else cfg.dropout
        self.rnn = nn.LSTM(
            input_size=input_size, hidden_size=cfg.hidden_size,
            num_layers=cfg.n_layers, dropout=dropout,
            bidirectional=cfg.birnn, batch_first=True)

    def forward(self, x, length, mask):

        # sort the lengths and remember the indices
        sorted_lengths, perm_index = length.sort(0, descending=True)

        # this allows us to put it back in the original order
        rev_perm_index = torch.zeros_like(perm_index)
        for i, sort_pos in enumerate(perm_index):
            rev_perm_index[sort_pos] = i

        # sort the input word embeddings x
        sorted_x = torch.index_select(x, 0, perm_index)

        # pack sorted embeddings, run RNN, unpack
        pack_x = pack_padded_sequence(sorted_x, sorted_lengths,
                                      batch_first=True)
        packed_outputs, (ht, ct) = self.rnn(pack_x)
        outputs, _ = pad_packed_sequence(packed_outputs,
                                         batch_first=True,
                                         padding_value=self.cfg.pad_idx)

        # use final states
        ht = torch.cat([ht[-2], ht[-1]], dim=-1)  # [B, T, 2D]
        final = torch.index_select(ht, 0, rev_perm_index).contiguous()

        return final
