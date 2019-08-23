import math
import torch
from torch import nn


class RCNNCell(nn.Module):
    """
    RCNN Cell
    Used in "Rationalizing Neural Predictions" (Lei et al., 2016)
    This is the bigram version of the cell.
    """

    def __init__(self, input_size, hidden_size):
        """
        Initializer.
        :param input_size:
        :param hidden_size:
        """
        super(RCNNCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # project input for λt, ct1, ct2
        self.ih_layer = nn.Linear(input_size, 3 * hidden_size, bias=False)

        # project previous state for λt (and add bias)
        self.hh_layer = nn.Linear(hidden_size, hidden_size, bias=True)

        # final output bias
        self.bias = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        """This is PyTorch's default initialization method"""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_, prev, mask=None):
        """
        input is (batch, input_size)
        hx is ((batch, hidden_size), (batch, hidden_size))
        """
        prev_h, prev_c1, prev_c2 = prev

        # project input x and previous state h
        ih_combined = self.ih_layer(input_)
        wlx, w1x, w2x = torch.chunk(ih_combined, 3, dim=-1)
        ulh = self.hh_layer(prev_h)

        # main RCNN computation
        lambda_ = (wlx + ulh).sigmoid()
        c1 = lambda_ * prev_c1 + (1 - lambda_) * w1x
        c2 = lambda_ * prev_c2 + (1 - lambda_) * (prev_c1 + w2x)

        h = (c2 + self.bias).tanh()

        return h, c1, c2

    def __repr__(self):
        return "{}({:d}, {:d})".format(
            self.__class__.__name__, self.input_size, self.hidden_size)


class RCNN(nn.Module):
    """
    Encodes sentence with an RCNN
    Assumes batch-major tensors.
    """

    def __init__(self, in_features, hidden_size, bidirectional=False):
        super(RCNN, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.cell = RCNNCell(in_features, hidden_size)

        if bidirectional:
            self.cell_rev = RCNNCell(in_features, hidden_size)
        else:
            self.cell_rev = None

    @staticmethod
    def empty_state(batch_size, hidden_size, device):
        """
        Returns an initial empty state.
        :param batch_size:
        :param hidden_size:
        :param device:
        :return: tuple of (h, c1, c2)
        """
        h_prev = torch.zeros(batch_size, hidden_size, device=device)
        c1_prev = torch.zeros(batch_size, hidden_size, device=device)
        c2_prev = torch.zeros(batch_size, hidden_size, device=device)
        state = (h_prev, c1_prev, c2_prev)
        return state

    @staticmethod
    def _step(x_t, cell, state: tuple, mask_t):
        """
        Take a single step.
        :param x: the input for this time step [B, D]
        :param state: tuple of (h, c1, c2)
        :param mask_t: mask for this time step [B]
        :return:
        """
        h_prev, c1_prev, c2_prev = state
        mask_t = mask_t.unsqueeze(-1)

        h, c1, c2 = cell(x_t, state)  # step

        h_prev = mask_t * h + (1 - mask_t) * h_prev
        c1_prev = mask_t * c1 + (1 - mask_t) * c1_prev
        c2_prev = mask_t * c2 + (1 - mask_t) * c2_prev

        state = (h_prev, c1_prev, c2_prev)
        return state

    @staticmethod
    def _unroll(x, cell, mask,
                state: tuple = None):

        batch_size, time, emb_size = x.size()
        assert mask.size(1) == time, "time mask mismatch"

        # initial state
        if state is None:
            state = RCNN.empty_state(
                batch_size, cell.hidden_size, device=x.device)

        # process this time-major
        x = x.transpose(0, 1).contiguous()                # [T, B, D]
        mask = mask.transpose(0, 1).contiguous().float()  # time-major: [T, B]

        # process input x one time step at a time
        outputs = []

        for x_t, mask_t in zip(x, mask):
            # only update if mask active (skip zeroed words)
            state = RCNN._step(x_t, cell, state, mask_t)
            outputs.append(state[0])

        # return batch-major
        outputs = torch.stack(outputs, dim=1)  # [batch_size, time, D]

        return outputs

    def forward(self, x, mask, lengths=None, state: tuple = None):
        """
        :param x: input sequence [B, T, D] (batch-major)
        :param mask: mask with 0s for invalid positions
        :param lengths:
        :param state: take a step from this state, or None to start from zeros
        :return:
        """
        assert lengths is not None, "provide lengths"

        # forward pass
        outputs = RCNN._unroll(x, self.cell, mask, state=state)

        # only if this is a full unroll (full sequence, e.g. for encoder)
        # extract final states from forward outputs
        batch_size, time, dim = outputs.size()

        final_indices = torch.arange(batch_size).to(x.device)
        final_indices = (final_indices * time) + lengths - 1
        final_indices = final_indices.long()

        final = outputs.view([-1, dim]).index_select(0, final_indices)

        if self.bidirectional:
            assert state is None, \
                "can only provide state for unidirectional RCNN"

            # backward pass
            idx_rev = torch.arange(x.size(1) - 1, -1, -1)
            mask_rev = mask[:, idx_rev]  # fix for pytorch 1.2
            x_rev = x[:, idx_rev]
            outputs_rev = RCNN._unroll(x_rev, self.cell_rev, mask_rev)
            final_rev = outputs_rev[:, -1, :].squeeze(1)
            # outputs_rev = outputs_rev.flip(1)
            outputs_rev = outputs_rev[:, idx_rev]  # back into original order

            # concatenate with forward pass
            final = torch.cat([final, final_rev], dim=-1)
            outputs = torch.cat([outputs, outputs_rev], dim=-1)

        # mask out invalid positions
        outputs = torch.where(mask.unsqueeze(2), outputs, x.new_zeros([1]))

        return outputs, final
