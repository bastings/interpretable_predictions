
from latent_rationale.snli.models.recurrent import RecurrentModel
from latent_rationale.snli.models.decomposable import DecompAttModel
from latent_rationale.snli.models.decomposable_kuma import KumaDecompAttModel


def build_model(cfg, vocab):
    """
    Build model according to config.
    :param cfg:
    :param vocab:
    :return:
    """

    if cfg.model == "decomposable":
        if "kuma" in cfg.dist:
            model = KumaDecompAttModel(cfg, vocab)
        else:
            model = DecompAttModel(cfg, vocab)

    elif cfg.model == "recurrent":
        model = RecurrentModel(cfg, vocab)
    else:
        raise ValueError("unknown model: %s" % cfg.model)

    return model
