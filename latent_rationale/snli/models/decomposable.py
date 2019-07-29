import torch
import torch.nn as nn

from torch.nn import ReLU, Dropout, Linear, Embedding
from latent_rationale.nn.attention import DeepDotAttention
from latent_rationale.nn.position import get_relative_positions
from latent_rationale.snli.util import masked_softmax, get_z_counts


class DecompAttModel(nn.Module):
    """
    Decomposable Attention model
    """

    def __init__(self, cfg, vocab):
        super(DecompAttModel, self).__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.n_embed, cfg.embed_size,
                                  padding_idx=cfg.pad_idx)
        self.vocab = vocab
        self.pad_idx = cfg.pad_idx
        self.dist_type = cfg.dist if cfg.dist else ""
        self.use_self_attention = cfg.self_attention
        self.selection = cfg.selection

        inp_dim = cfg.embed_size
        dim = cfg.hidden_size

        if cfg.fix_emb:
            self.embed.weight.requires_grad = False

        if self.cfg.projection:
            self.projection = nn.Linear(cfg.embed_size, cfg.proj_size)
            inp_dim = cfg.proj_size

        self.dropout = Dropout(p=cfg.dropout)
        self.activation = ReLU()

        if cfg.self_attention and not cfg.transformer:
            self.max_dist = 11
            self.dist_embed = Embedding(2 * self.max_dist + 1, 1)
            self.self_attention = DeepDotAttention(
                inp_dim, dim, dropout=cfg.dropout)

            inp_dim = inp_dim * 2
            self.self_att_dropout = Dropout(p=cfg.dropout)

        # set attention mechanism (between premise and hypothesis)
        self.attention = DeepDotAttention(inp_dim, dim, dropout=cfg.dropout)

        self.compare_layer = nn.Sequential(
            Linear(inp_dim * 2, dim), self.activation, self.dropout,
            Linear(dim, dim), self.activation, self.dropout
        )

        self.aggregate_layer = nn.Sequential(
            Linear(dim * 2, dim), self.activation, self.dropout,
            Linear(dim, dim), self.activation, self.dropout
        )

        self.output_layer = Linear(dim, cfg.output_size, bias=False)

        self.lagrange = nn.Parameter(torch.zeros([1]))

        # for extracting attention
        self.hypo_mask = None
        self.prem_mask = None
        self.prem2hypo_att = None
        self.hypo2prem_att = None
        self.prem_self_att = None
        self.hypo_self_att = None

        self.mask_diagonal = cfg.mask_diagonal
        self.relu_projection = False
        self.use_self_att_dropout = False

        self.prem_self_att_samples = []
        self.hypo_self_att_samples = []

        self.reset_params()
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def reset_params(self):
        """Custom initialization"""

        with torch.no_grad():
            for name, p in self.named_parameters():

                if "embed" in name:
                    continue
                else:
                    if p.dim() > 1:
                        gain = 1.
                        # print("xavier", name, p.size(), "gain=", gain)
                        nn.init.xavier_uniform_(p, gain=gain)
                    else:
                        # print("zeros ", name, p.size())
                        nn.init.zeros_(p)

        if hasattr(self, "dist_embed"):
            std = 1.0
            # print("Distance parameters init with normal =", std)
            with torch.no_grad():
                self.dist_embed.weight.normal_(mean=0, std=std)

    def _add_rel_dists(self, x):
        """add matrix of relative distances"""
        rel_dists = get_relative_positions(
            x.size(1), self.max_dist, device=x.device)
        rel_dists = self.dist_embed(rel_dists).squeeze(-1).unsqueeze(0)
        return x + rel_dists

    def _mask_diagonal(self, x):
        """block the diagonal so a word does not self-align"""
        eye = torch.eye(x.size(1), dtype=torch.uint8, device=x.device)
        return torch.where(eye, x.new_full(x.size(), float('-inf')), x)

    def _mask_padding(self, x, mask, value=0.):
        """
        Mask should be true/1 for valid positions, false/0 for invalid ones.
        :param x:
        :param mask:
        :return:
        """
        return torch.where(mask, x, x.new_full([1], value))

    def forward(self, batch):

        prem_input, prem_lengths = batch.premise
        hypo_input, hypo_lengths = batch.hypothesis

        self.prem_mask = prem_mask = (prem_input != self.pad_idx)
        self.hypo_mask = hypo_mask = (hypo_input != self.pad_idx)

        prem = self.embed(prem_input)
        hypo = self.embed(hypo_input)

        # do not backpropagate through embeddings when fixed
        if self.cfg.fix_emb:
            prem = prem.detach()
            hypo = hypo.detach()

        # project embeddings (unless disabled)
        if self.cfg.projection:
            hypo = self.projection(hypo)
            prem = self.projection(prem)

            if self.relu_projection:
                hypo = self.activation(hypo)
                prem = self.activation(prem)

            hypo = self.dropout(hypo)
            prem = self.dropout(prem)

        # this is the original self-attention from DA
        if self.cfg.self_attention and not self.transformer:

            # self-attention (self dot product)
            prem_self_att = self.self_attention(prem, prem)
            hypo_self_att = self.self_attention(hypo, hypo)

            # add relative distances
            prem_self_att = self._add_rel_dists(prem_self_att)
            hypo_self_att = self._add_rel_dists(hypo_self_att)

            if self.mask_diagonal:
                prem_self_att = self._mask_diagonal(prem_self_att)
                hypo_self_att = self._mask_diagonal(hypo_self_att)

            prem_self_att = masked_softmax(prem_self_att, prem_mask.unsqueeze(1))
            hypo_self_att = masked_softmax(hypo_self_att, hypo_mask.unsqueeze(1))

            self.prem_self_att = prem_self_att
            self.hypo_self_att = hypo_self_att

            prem_self_att_ctx = prem_self_att @ prem
            hypo_self_att_ctx = hypo_self_att @ hypo

            if self.use_self_att_dropout:
                prem_self_att_ctx = self.self_att_dropout(prem_self_att_ctx)
                hypo_self_att_ctx = self.self_att_dropout(hypo_self_att_ctx)

            prem = torch.cat([prem, prem_self_att_ctx], dim=-1)
            hypo = torch.cat([hypo, hypo_self_att_ctx], dim=-1)

        # compute attention
        sim = self.attention(prem, hypo)  # [B, M, N]

        self.prem2hypo_att = masked_softmax(sim, hypo_mask.unsqueeze(1))  # [B, M, N]
        self.hypo2prem_att = masked_softmax(
            sim.transpose(1, 2).contiguous(), prem_mask.unsqueeze(1))  # [B, N, M]

        # take weighed sum of hypo (premise) based on attention weights
        attended_hypo = self.prem2hypo_att @ hypo
        attended_prem = self.hypo2prem_att @ prem

        # compare input
        prem_compared = self.compare_layer(
            torch.cat([prem, attended_hypo], dim=-1))
        hypo_compared = self.compare_layer(
            torch.cat([hypo, attended_prem], dim=-1))

        prem_compared = prem_compared * prem_mask.float().unsqueeze(-1)
        hypo_compared = hypo_compared * hypo_mask.float().unsqueeze(-1)

        prem_compared = prem_compared.sum(dim=1)
        hypo_compared = hypo_compared.sum(dim=1)

        aggregate = self.aggregate_layer(
            torch.cat([prem_compared, hypo_compared], dim=-1))

        scores = self.output_layer(aggregate)

        return scores

    def get_loss(self, logits, targets):

        optional = {}
        batch_size = logits.size(0)
        loss = self.criterion(logits, targets) / batch_size
        optional["ce"] = loss.item()

        return loss, optional

