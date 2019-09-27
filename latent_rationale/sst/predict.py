import os
from collections import OrderedDict

import torch
import torch.optim

from latent_rationale.sst.vocabulary import Vocabulary
from latent_rationale.sst.models.model_helpers import build_model
from latent_rationale.sst.util import get_predict_args, sst_reader, \
    load_glove, print_parameters, get_device, find_ckpt_in_directory, \
    plot_dataset
from latent_rationale.sst.evaluate import evaluate


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
    cfg = ckpt["cfg"]

    for k, v in cfg.items():
        print("{:20} : {:10}".format(k, v))

    batch_size = cfg.get("eval_batch_size", 25)

    # Let's load the data into memory.
    train_data = list(sst_reader("data/sst/train.txt"))
    dev_data = list(sst_reader("data/sst/dev.txt"))
    test_data = list(sst_reader("data/sst/test.txt"))

    print("train", len(train_data))
    print("dev", len(dev_data))
    print("test", len(test_data))

    example = dev_data[0]
    print("First train example:", example)
    print("First train example tokens:", example.tokens)
    print("First train example label:", example.label)

    vocab = Vocabulary()
    vectors = load_glove(cfg["word_vectors"], vocab)  # this populates vocab

    # Map the sentiment labels 0-4 to a more readable form (and the opposite)
    i2t = ["very negative", "negative", "neutral", "positive", "very positive"]
    t2i = OrderedDict({p: i for p, i in zip(i2t, range(len(i2t)))})

    # Build model
    model = build_model(cfg["model"], vocab, t2i, cfg)

    # load parameters from checkpoint into model
    print("Loading saved model..")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    print("Done")

    # print model
    print(model)
    print_parameters(model)

    print("Evaluating")

    dev_eval = evaluate(model, dev_data, batch_size=batch_size, device=device)
    print("dev acc", dev_eval["acc"])

    test_eval = evaluate(model, test_data, batch_size=batch_size, device=device)
    print("test acc", test_eval["acc"])

    print("Plotting attention scores")
    if predict_cfg.plot:
        plot_save_path = os.path.join(cfg["save_path"], "plots")
        if not os.path.exists(plot_save_path):
            os.makedirs(plot_save_path)
        plot_dataset(model, dev_data, batch_size=batch_size,
                     device=device, save_path=plot_save_path)


if __name__ == "__main__":
    predict()
