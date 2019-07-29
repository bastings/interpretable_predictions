import torch
import torch.optim
import torch.nn as nn

import os
import numpy as np
from torchtext import data
from latent_rationale.snli.text import SNLI
from latent_rationale.snli.models.model_helper import build_model
from latent_rationale.snli.util import print_config, print_parameters, get_device, get_data_fields, \
    load_glove_words, get_predict_args, find_ckpt_in_directory
from latent_rationale.common.util import make_kv_string
from latent_rationale.snli.util import print_examples, extract_attention
from latent_rationale.snli.evaluate import evaluate


def predict():

    predict_cfg = get_predict_args()
    device = get_device()
    print(device)

    # load checkpoint
    ckpt_path = find_ckpt_in_directory(predict_cfg.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]

    # to know which words to UNK we need to know the Glove vocabulary
    glove_words = load_glove_words(cfg.word_vectors)

    # load data sets
    print("Loading data... ", end="")
    input_field, label_field, not_in_glove = get_data_fields(glove_words)
    train_data, dev_data, test_data = SNLI.splits(input_field, label_field)
    print("Done")
    print("Words not in glove:", len(not_in_glove))

    # build vocabulary (deterministic so no need to load it)
    input_field.build_vocab(train_data, dev_data, test_data,
                            vectors=None, vectors_cache=None)
    label_field.build_vocab(train_data)

    # construct model
    model = build_model(cfg, input_field.vocab)

    # load parameters from checkpoint into model
    print("Loading saved model..")
    model.load_state_dict(ckpt["model"])
    print("Done")

    train_iter = data.BucketIterator(
        train_data, batch_size=cfg.batch_size, train=False, repeat=False,
        device=device if torch.cuda.is_available() else -1)

    dev_iter = data.BucketIterator(
        dev_data, batch_size=cfg.batch_size, train=False, repeat=False,
        device=device if torch.cuda.is_available() else -1)

    test_iter = data.BucketIterator(
        test_data, batch_size=cfg.batch_size, train=False, repeat=False,
        device=device if torch.cuda.is_available() else -1)

    print_config(cfg)

    print("Embedding variance:", torch.var(model.embed.weight).item())
    model.to(device)

    print_parameters(model)
    print(model)

    # switch model to evaluation mode
    model.eval()
    train_iter.init_epoch()
    dev_iter.init_epoch()
    test_iter.init_epoch()

    criterion = nn.CrossEntropyLoss(reduction='sum')

    print("Starting evaluation..")
    eval_list = [("train", train_iter), ("dev", dev_iter), ("test", test_iter)]
    for name, it in eval_list:
        eval_result = evaluate(model, criterion, it)
        eval_str = make_kv_string(eval_result)
        print("# Evaluation {}: {}".format(name, eval_str))

    # print dev examples for highscore
    dev_iter.init_epoch()
    p2h, h2p, prems, hypos, predictions, targets = extract_attention(
        model, dev_iter, input_field.vocab, label_field.vocab)
    np.savez(os.path.join(cfg.save_path, "dev_items"),
             p2h=p2h, h2p=h2p, prems=prems, hypos=hypos,
             predictions=predictions, targets=targets)

    # print dev examples for highscore
    dev_iter.init_epoch()
    dev_dir = os.path.join(cfg.save_path, "dev")
    if not os.path.exists(dev_dir):
        os.makedirs(dev_dir)
    print_examples(model, dev_iter, input_field.vocab, label_field.vocab,
                   dev_dir, 0, n=-1)


if __name__ == "__main__":
    predict()
