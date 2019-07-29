import os
import torch
import torch.optim

from latent_rationale.beer.vocabulary import Vocabulary
from latent_rationale.beer.models.model_helpers import build_model
from latent_rationale.beer.util import \
    print_parameters, get_predict_args, get_device, find_ckpt_in_directory, \
    beer_annotations_reader, beer_reader, load_embeddings
from latent_rationale.beer.evaluate import evaluate_loss, evaluate_rationale
from latent_rationale.common.util import make_kv_string


def predict():
    """
    Make predictions with a saved model.
    """

    predict_cfg = get_predict_args()
    device = get_device()
    print(device)

    # load checkpoint
    ckpt_path = find_ckpt_in_directory(predict_cfg.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device)
    best_iter = ckpt["best_iter"]
    cfg = ckpt["cfg"]
    aspect = cfg["aspect"]

    for k, v in cfg.items():
        print("{:20} : {:10}".format(k, str(v)))

    eval_batch_size = 64

    print("Loading data")
    dev_data = list(beer_reader(cfg["dev_path"]))
    test_data = beer_annotations_reader(cfg["test_path"], aspect=aspect)

    print("dev", len(dev_data))
    print("test", len(test_data))

    print("Loading pre-trained word embeddings")
    vocab = Vocabulary()
    vectors = load_embeddings(cfg["embeddings"], vocab)  # required for vocab

    # build model
    model = build_model(cfg["model"], vocab, cfg=cfg)

    # load parameters from checkpoint into model
    print("Loading saved model..")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    print("Done")

    print(model)
    print_parameters(model)

    print("Evaluating")
    dev_eval = evaluate_loss(
        model, dev_data, batch_size=eval_batch_size,
        device=device, cfg=cfg)
    test_eval = evaluate_loss(
        model, test_data, batch_size=eval_batch_size,
        device=device, cfg=cfg)

    if hasattr(model, "z"):
        path = os.path.join(
            cfg["save_path"], "final_rationales.txt")
        test_precision, test_macro_prec = evaluate_rationale(
            model, test_data, aspect=aspect, device=device,
            batch_size=eval_batch_size, path=path)
    else:
        test_precision = 0.
        test_macro_prec = 0.
    test_eval["precision"] = test_precision
    test_eval["macro_precision"] = test_macro_prec

    dev_s = make_kv_string(dev_eval)
    test_s = make_kv_string(test_eval)

    print("best model iter {:d} dev {} test {}".format(
        best_iter, dev_s, test_s))


if __name__ == "__main__":
    predict()
