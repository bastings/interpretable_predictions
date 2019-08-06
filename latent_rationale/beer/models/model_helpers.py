#!/usr/bin/env python

from latent_rationale.beer.models.simpleclassifier import SimpleClassifier
from latent_rationale.beer.models.latent import LatentRationaleModel
from latent_rationale.beer.models.rl import RLModel


def build_model(model_type, vocab, cfg=None):

    aspect = cfg["aspect"]
    emb_size = cfg["emb_size"]
    hidden_size = cfg["hidden_size"]
    dropout = cfg["dropout"]
    layer = cfg["layer"]
    vocab_size = len(vocab.w2i)
    dependent_z = cfg["dependent_z"]

    if aspect > -1:
        output_size = 1
    else:
        output_size = 5

    if model_type == "baseline":
        return SimpleClassifier(
            vocab_size, emb_size, hidden_size, output_size,
            vocab=vocab, dropout=dropout, layer=layer)
    elif model_type == "rl":
        sparsity = cfg["sparsity"]
        coherence = cfg["coherence"]

        return RLModel(
            vocab_size=vocab_size, emb_size=emb_size, hidden_size=hidden_size,
            output_size=output_size, vocab=vocab, dropout=dropout,
            dependent_z=dependent_z, layer=layer,
            sparsity=sparsity, coherence=coherence)
    elif model_type == "latent":
        selection = cfg["selection"]
        lasso = cfg["lasso"]
        lagrange_alpha = cfg["lagrange_alpha"]
        lagrange_lr = cfg["lagrange_lr"]
        lambda_init = cfg["lambda_init"]
        lambda_min = cfg["lambda_min"]
        lambda_max = cfg["lambda_max"]
        return LatentRationaleModel(
            vocab_size=vocab_size, emb_size=emb_size, hidden_size=hidden_size,
            output_size=output_size, vocab=vocab, dropout=dropout,
            dependent_z=dependent_z, layer=layer,
            selection=selection, lasso=lasso,
            lagrange_alpha=lagrange_alpha, lagrange_lr=lagrange_lr,
            lambda_init=lambda_init,
            lambda_min=lambda_min, lambda_max=lambda_max)
    else:
        raise ValueError("Unknown model")
