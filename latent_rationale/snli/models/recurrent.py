import torch
import torch.nn as nn
from latent_rationale.snli.encoder import RecurrentEncoder


class RecurrentModel(nn.Module):

    def __init__(self, cfg, vocab):
        super(RecurrentModel, self).__init__()
        self.config = cfg
        self.embed = nn.Embedding(cfg.n_embed, cfg.embed_size,
                                  padding_idx=cfg.pad_idx)
        self.vocab = vocab
        self.pad_idx = cfg.pad_idx

        if cfg.fix_emb:
            self.embed.weight.requires_grad = False

        if self.config.projection:
            self.projection = nn.Linear(cfg.embed_size, cfg.proj_size)

        self.encoder = RecurrentEncoder(cfg)
        self.dropout = nn.Dropout(p=cfg.dropout)
        self.dropout = nn.Dropout(p=cfg.dropout)
        self.activation = nn.ReLU()

        inp_dim = 2 * cfg.hidden_size
        inp_dim = inp_dim * 2  # fancy input

        if self.config.birnn:
            inp_dim *= 2

        self.out = nn.Sequential(
            nn.Linear(inp_dim, inp_dim),
            self.activation,
            self.dropout,
            nn.Linear(inp_dim, cfg.output_size, bias=False))

        self.criterion = nn.CrossEntropyLoss(reduction='sum')


    def forward(self, batch):

        prem_input, prem_lengths = batch.premise
        hypo_input, hypo_lengths = batch.hypothesis

        prem_mask = (prem_input != self.pad_idx)
        hypo_mask = (hypo_input != self.pad_idx)

        prem_embed = self.embed(prem_input)
        hypo_embed = self.embed(hypo_input)

        # do not backpropagate through embeddings when fixed
        if self.config.fix_emb:
            prem_embed = prem_embed.detach()
            hypo_embed = hypo_embed.detach()

        # project embeddings (unless disabled)
        if self.config.projection:
            prem_embed = self.activation(self.projection(prem_embed))
            hypo_embed = self.activation(self.projection(hypo_embed))

        premise = self.encoder(prem_embed, prem_lengths, prem_mask)
        hypothesis = self.encoder(hypo_embed, hypo_lengths, hypo_mask)

        # fancy combination of hypothesis and premise from SPINN
        fancy_combination = torch.cat(
            [premise,
             hypothesis,
             torch.abs(premise-hypothesis),
             premise * hypothesis], -1)

        scores = self.out(fancy_combination)
        return scores

    def get_loss(self, logits, targets):
        optional = {}
        batch_size = logits.size(0)
        loss = self.criterion(logits, targets) / batch_size
        optional["ce"] = loss.item()

        return loss, optional


