from collections import OrderedDict
import torch
import torch.nn as nn

from torch.nn import ReLU, Dropout, Linear, Embedding
from latent_rationale.nn.kuma_attention import KumaAttention
from latent_rationale.nn.attention import DeepDotAttention
from latent_rationale.nn.position import get_relative_positions
from latent_rationale.snli.util import masked_softmax, get_z_counts


class KumaDecompAttModel(nn.Module):
    """
    Decomposable Attention model with Kuma attention
    """

    def __init__(self, cfg, vocab):
        super(KumaDecompAttModel, self).__init__()
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

        if cfg.self_attention:
            self.max_dist = 11
            self.dist_embed = Embedding(2 * self.max_dist + 1, 1)
            self.self_attention = DeepDotAttention(
                inp_dim, dim, dropout=cfg.dropout)

            inp_dim = inp_dim * 2
            self.self_att_dropout = Dropout(p=cfg.dropout)

        # set attention mechanism (between premise and hypothesis)
        if "kuma" in self.dist_type:
            self.attention = KumaAttention(
                inp_dim, dim, dropout=cfg.dropout, dist_type=self.dist_type)
        else:
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

        # lagrange (for controlling HardKuma attention percentage)
        self.lagrange_lr = cfg.lagrange_lr
        self.lagrange_alpha = cfg.lagrange_alpha
        self.lambda_init = cfg.lambda_init
        self.register_buffer('lambda0', torch.full((1,), self.lambda_init))
        self.register_buffer('c0_ma', torch.full((1,), 0.))  # moving average

        # for extracting attention
        self.hypo_mask = None
        self.prem_mask = None
        self.prem2hypo_att = None
        self.hypo2prem_att = None
        self.prem_self_att = None
        self.hypo_self_att = None
        self.prem_self_att_dist = None
        self.hypo_self_att_dist = None

        self.mask_diagonal = cfg.mask_diagonal
        self.relu_projection = False
        self.use_self_att_dropout = False

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

        if self.cfg.self_attention:

            # self-attention (self dot product)
            prem_self_att = self.self_attention(prem, prem)
            self.prem_self_att_dist = self.self_attention.dist

            hypo_self_att = self.self_attention(hypo, hypo)
            self.hypo_self_att_dist = self.self_attention.dist

            # add relative distances
            prem_self_att = self._add_rel_dists(prem_self_att)
            hypo_self_att = self._add_rel_dists(hypo_self_att)

            if self.mask_diagonal:
                prem_self_att = self._mask_diagonal(prem_self_att)
                hypo_self_att = self._mask_diagonal(hypo_self_att)

            # mask
            prem_self_att = torch.where(self.prem_mask.unsqueeze(1),
                                        prem_self_att, prem.new_zeros([1]))
            hypo_self_att = torch.where(self.hypo_mask.unsqueeze(1),
                                        hypo_self_att, hypo.new_zeros([1]))

            prem_self_att = masked_softmax(prem_self_att,
                                           prem_mask.unsqueeze(1))
            hypo_self_att = masked_softmax(hypo_self_att,
                                           hypo_mask.unsqueeze(1))

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

        if self.dist_type is not None and self.dist_type == "hardkuma":

            # mask invalid attention positions (note: it is symmetric here!)
            sim = self._mask_padding(sim, hypo_mask.unsqueeze(1), 0.)
            sim = self._mask_padding(sim, prem_mask.unsqueeze(2), 0.)

            self.prem2hypo_att = sim
            self.hypo2prem_att = sim.transpose(1, 2)

        else:
            prem2hypo_att = sim
            hypo2prem_att = sim.transpose(1, 2)

            self.prem2hypo_att = masked_softmax(
                prem2hypo_att, hypo_mask.unsqueeze(1))  # [B, |p|, |h|]
            self.hypo2prem_att = masked_softmax(
                hypo2prem_att, prem_mask.unsqueeze(1))  # [B, |h|, |p|]

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

        optional = OrderedDict()

        batch_size = logits.size(0)

        loss = self.criterion(logits, targets) / batch_size
        optional["ce"] = loss.item()

        # training stats
        if self.training:
            # note that the attention matrix is symmetric now, so we only
            # need to compute the counts for prem2hypo
            z0, zc, z1 = get_z_counts(
                self.prem2hypo_att, self.prem_mask, self.hypo_mask)
            zt = float(z0 + zc + z1)
            optional["p2h_0"] = z0 / zt
            optional["p2h_c"] = zc / zt
            optional["p2h_1"] = z1 / zt
            optional["p2h_selected"] = 1 - optional["p2h_0"]

        # regularize sparsity
        assert isinstance(self.attention, KumaAttention), \
            "expected HK attention for this model, please set dist=hardkuma"

        if self.selection > 0:

            # Kuma attention distribution (computed in forward call)
            z_dist = self.attention.dist

            # pre-compute pdf(0)  shape: [B, |prem|, |hypo|]
            pdf0 = z_dist.pdf(0.)
            pdf0 = pdf0.squeeze(-1)

            prem_lengths = self.prem_mask.sum(1).float()
            hypo_lengths = self.hypo_mask.sum(1).float()

            # L0 regularizer

            # probability of being non-zero (masked for invalid positions)
            # we first mask all invalid positions in the tensor
            # first we mask invalid hypothesis positions
            #   (dim 2,broadcast over dim1)
            # then we mask invalid premise positions
            #   (dim 1, broadcast over dim 2)
            pdf_nonzero = 1. - pdf0  # [B, T]
            pdf_nonzero = self._mask_padding(
                pdf_nonzero, mask=self.hypo_mask.unsqueeze(1), value=0.)
            pdf_nonzero = self._mask_padding(
                pdf_nonzero, mask=self.prem_mask.unsqueeze(2), value=0.)

            l0 = pdf_nonzero.sum(2) / (hypo_lengths.unsqueeze(1) + 1e-9)
            l0 = l0.sum(1) / (prem_lengths + 1e-9)
            l0 = l0.sum() / batch_size

            # lagrange dissatisfaction, batch average of the constraint
            c0_hat = (l0 - self.selection)

            # moving average of the constraint
            self.c0_ma = self.lagrange_alpha * self.c0_ma + (
                    1 - self.lagrange_alpha) * c0_hat.item()

            # compute smoothed constraint (equals moving average c0_ma)
            c0 = c0_hat + (self.c0_ma.detach() - c0_hat.detach())

            # update lambda
            self.lambda0 = self.lambda0 * torch.exp(
                self.lagrange_lr * c0.detach())

            with torch.no_grad():
                optional["cost0_l0"] = l0.item()
                optional["target0"] = self.selection
                optional["c0_hat"] = c0_hat.item()
                optional["c0"] = c0.item()  # same as moving average
                optional["lambda0"] = self.lambda0.item()
                optional["lagrangian0"] = (self.lambda0 * c0_hat).item()
                optional["a"] = z_dist.a.mean().item()
                optional["b"] = z_dist.b.mean().item()

            loss = loss + self.lambda0.detach() * c0

        return loss, optional
