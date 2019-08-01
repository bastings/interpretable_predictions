#!/usr/bin/env python

from latent_rationale.sst.models.baseline import Baseline
from latent_rationale.sst.models.rl import RLModel
from latent_rationale.sst.models.latent import LatentRationaleModel


def build_model(model_type, vocab, t2i, cfg):

    vocab_size = len(vocab.w2i)
    output_size = len(t2i)

    emb_size = cfg["embed_size"]
    hidden_size = cfg["hidden_size"]
    dropout = cfg["dropout"]
    layer = cfg["layer"]
    dependent_z = cfg.get("dependent_z", False)

    selection = cfg["selection"]
    lasso = cfg["lasso"]

    sparsity = cfg["sparsity"]
    coherence = cfg["coherence"]

    assert 0 < selection <= 1.0, "selection must be in (0, 1]"

    if model_type == "baseline":
        return Baseline(
            vocab_size, emb_size, hidden_size, output_size, vocab=vocab,
            dropout=dropout, layer=layer)
    elif model_type == "rl":
        return RLModel(
            vocab_size=vocab_size, emb_size=emb_size,
            hidden_size=hidden_size, output_size=output_size,
            vocab=vocab, dropout=dropout, layer=layer,
            dependent_z=dependent_z,
            sparsity=sparsity, coherence=coherence)
    elif model_type == "latent":
        lambda_init = cfg["lambda_init"]
        lagrange_lr = cfg["lagrange_lr"]
        lagrange_alpha = cfg["lagrange_alpha"]
        return LatentRationaleModel(
            vocab_size=vocab_size, emb_size=emb_size,
            hidden_size=hidden_size, output_size=output_size,
            vocab=vocab, dropout=dropout, layer=layer,
            dependent_z=dependent_z,
            selection=selection, lasso=lasso,
            lambda_init=lambda_init,
            lagrange_lr=lagrange_lr, lagrange_alpha=lagrange_alpha)
    else:
        raise ValueError("Unknown model")
