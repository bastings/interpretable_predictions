import torch
from torch import nn
import numpy as np

from latent_rationale.common.util import get_encoder
from latent_rationale.nn.bernoulli_gate import BernoulliGate
from latent_rationale.nn.rcnn import RCNNCell


class IndependentGenerator(nn.Module):
    """
    The Generator takes an input text and returns samples from p(z|x)
    """

    def __init__(self,
                 embed:       nn.Embedding = None,
                 hidden_size: int = 200,
                 dropout:     float = 0.1,
                 layer:       str = "rcnn"
                 ):

        super(IndependentGenerator, self).__init__()

        emb_size = embed.weight.shape[1]
        enc_size = hidden_size * 2

        self.embed_layer = nn.Sequential(embed, nn.Dropout(p=dropout))
        self.enc_layer = get_encoder(layer, emb_size, hidden_size)

        self.z_layer = BernoulliGate(enc_size)

        self.z = None      # z samples
        self.z_dists = []  # z distribution(s)

        self.report_params()

    def report_params(self):
        count = 0
        for name, p in self.named_parameters():
            if p.requires_grad and "embed" not in name:
                count += np.prod(list(p.shape))
        print("{} #params: {}".format(self.__class__.__name__, count))

    def forward(self, x, mask):

        # encode sentence
        lengths = mask.long().sum(1)
        emb = self.embed_layer(x)  # [B, T, E]
        h, _ = self.enc_layer(emb, mask, lengths)

        # compute parameters for Bernoulli p(z|x)
        z_dist = self.z_layer(h)

        if self.training:  # sample
            z = z_dist.sample()  # [B, T, 1]
        else:  # deterministic
            z = (z_dist.probs >= 0.5).float()   # [B, T, 1]

        z = z.squeeze(-1)  # [B, T, 1]  -> [B, T]
        z = torch.where(mask, z, z.new_zeros([1]))

        self.z = z
        self.z_dists = [z_dist]

        return z


class DependentGenerator(nn.Module):
    """
    The Generator takes an input text and returns samples from p(z|x)
    """

    def __init__(self,
                 embed:       nn.Embedding = None,
                 hidden_size: int = 200,
                 dropout:     float = 0.1,
                 layer:       str = "rcnn",
                 z_rnn_size:  int = 30,
                 ):

        super(DependentGenerator, self).__init__()

        emb_size = embed.weight.shape[1]
        enc_size = hidden_size * 2

        self.embed_layer = nn.Sequential(embed, nn.Dropout(p=dropout))
        self.enc_layer = get_encoder(layer, emb_size, hidden_size)

        self.z_cell = RCNNCell(enc_size + 1, z_rnn_size)
        self.z_layer = BernoulliGate(enc_size + z_rnn_size)

        self.z = None      # z samples
        self.z_dists = []  # z distribution(s)

        self.report_params()

    def report_params(self):
        count = 0
        for name, p in self.named_parameters():
            if p.requires_grad and "embed" not in name:
                count += np.prod(list(p.shape))
        print("{} #params: {}".format(self.__class__.__name__, count))

    def forward(self, x, mask, num_samples=1):

        # encode sentence
        batch_size, time = x.size()
        lengths = mask.sum(1)
        emb = self.embed_layer(x)  # [B, T, E]
        h, _ = self.enc_layer(emb, mask, lengths)

        # predict z for each time step conditioning on previous z

        # repeat hidden states (for multiple samples for prediction)
        h = h.unsqueeze(1).repeat(1, num_samples, 1, 1)
        h = h.view(batch_size * num_samples, time, -1)
        h = h.transpose(0, 1)  # time, batch*num_samples, dim

        z = []
        z_dists = []

        # initial states  [1, B, z_rnn_dim]
        state = torch.zeros([3 * batch_size, self.z_cell.hidden_size],
                            device=x.device).chunk(3)

        for h_t, t in zip(h, range(time)):

            # compute Binomial z distribution for this time step
            z_t_dist = self.z_layer(torch.cat([h_t, state[0]], dim=-1))
            z_dists.append(z_t_dist)

            if self.training:
                # sample (once since we already repeated the state)
                z_t = z_t_dist.sample().detach()  # [B, 1]
            else:
                z_t = (z_t_dist.probs >= 0.5).float().detach()
            assert (z_t < 0.).sum().item() == 0, "cannot be smaller than 0."
            z.append(z_t)

            # update cell state (to make dependent decisions)
            rnn_input = torch.cat([h_t, z_t], dim=-1)  # [B, 2D+1]
            state = self.z_cell(rnn_input, state)

        z = torch.stack(z, dim=1).squeeze(-1)  # [B, T]
        z = torch.where(mask, z, z.new_zeros([1]))

        self.z = z
        self.z_dists = z_dists

        return z
