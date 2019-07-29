import os
import time
import torch
import torch.optim
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
import numpy as np
import shutil
from torch.utils.tensorboard import SummaryWriter

from torchtext import data

from latent_rationale.snli.text import SNLI
from latent_rationale.snli.constants import UNK_TOKEN, PAD_TOKEN, INIT_TOKEN
from latent_rationale.snli.models.model_helper import build_model
from latent_rationale.common.util import make_kv_string
from latent_rationale.snli.util import get_args, makedirs, print_examples, \
    print_config, print_parameters, load_glove_words, get_n_correct, \
    get_device, save_checkpoint, get_data_fields
from latent_rationale.snli.evaluate import evaluate


def train():
    """
    Main SNLI training loop.
    """
    cfg = get_args()

    # overwrite save_path or warn to specify another path
    if os.path.exists(cfg.save_path):
        if cfg.overwrite:
            shutil.rmtree(cfg.save_path)
        else:
            raise RuntimeError(
                "save_path already exists; specify a different path")

    makedirs(cfg.save_path)

    device = get_device()
    print("device:", device)

    writer = SummaryWriter(log_dir=cfg.save_path)  # TensorBoard

    print("Loading data... ", end="")
    glove_words = load_glove_words(cfg.word_vectors)
    input_field, label_field, not_in_glove = get_data_fields(glove_words)
    train_data, dev_data, test_data = SNLI.splits(input_field, label_field)
    print("Done")

    print("First train sentence:",
          "[prem]: " + " ".join(train_data[0].premise),
          "[hypo]: " + " ".join(train_data[0].hypothesis),
          "[lab]:  " + train_data[0].label, sep="\n", end="\n\n")

    # build vocabularies
    std = 1.
    input_field.build_vocab(
        train_data, dev_data, test_data,
        unk_init=lambda x: x.normal_(mean=0, std=std),
        vectors=cfg.word_vectors, vectors_cache=None)
    label_field.build_vocab(train_data)

    print("Words not in glove:", len(not_in_glove))

    cfg.n_embed = len(input_field.vocab)
    cfg.output_size = len(label_field.vocab)
    cfg.n_cells = cfg.n_layers
    cfg.pad_idx = input_field.vocab.stoi[PAD_TOKEN]
    cfg.unk_idx = input_field.vocab.stoi[UNK_TOKEN]
    cfg.init_idx = input_field.vocab.stoi[INIT_TOKEN]

    # normalize word embeddings (each word embedding has L2 norm of 1.)
    if cfg.normalize_embeddings:
        with torch.no_grad():
            input_field.vocab.vectors /= input_field.vocab.vectors.norm(
                2, dim=-1, keepdim=True)

    # zero out padding
    with torch.no_grad():
        input_field.vocab.vectors[cfg.pad_idx].zero_()

    # save vocabulary (not really needed but could be useful)
    with open(os.path.join(cfg.save_path, "vocab.txt"),
              mode="w", encoding="utf-8") as f:
        for t in input_field.vocab.itos:
            f.write(t + "\n")

    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train_data, dev_data, test_data), batch_size=cfg.batch_size,
        device=device)

    print_config(cfg)

    # double the number of cells for bidirectional networks
    if cfg.birnn:
        cfg.n_cells *= 2

    if cfg.resume_snapshot:
        ckpt = torch.load(cfg.resume_snapshot, map_location=device)
        cfg = ckpt["cfg"]
        model_state = ckpt["model"]

    # build model
    model = build_model(cfg, input_field.vocab)

    if cfg.resume_snapshot:
        model.load_state_dict(model_state)

    # load Glove word vectors
    if cfg.word_vectors:
        with torch.no_grad():
            model.embed.weight.data.copy_(input_field.vocab.vectors)

    model.to(device)

    print_parameters(model)
    print(model)

    trainable_parameters = list(filter(lambda p: p.requires_grad,
                                       model.parameters()))
    opt = Adam(trainable_parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler = ReduceLROnPlateau(opt, "max", patience=cfg.patience,
                                  factor=cfg.lr_decay, min_lr=cfg.min_lr,
                                  verbose=True)

    if cfg.eval_every == -1:
        cfg.eval_every = int(np.ceil(len(train_data) / cfg.batch_size))
        print("Eval every: %d" % cfg.eval_every)

    iterations = 0
    start = time.time()
    best_dev_acc = -1
    train_iter.repeat = False

    for epoch in range(cfg.epochs):
        train_iter.init_epoch()
        n_correct, n_total = 0, 0
        for batch_idx, batch in enumerate(train_iter):

            # switch model to training mode, clear gradient accumulators
            model.train()
            opt.zero_grad()

            iterations += 1

            # forward pass
            output = model(batch)

            # calculate accuracy of predictions in the current batch
            n_correct += get_n_correct(batch, output)
            n_total += batch.batch_size
            train_acc = 100. * n_correct / n_total

            # calculate loss of the network output with respect to train labels
            loss, optional = model.get_loss(output, batch.label)

            # backpropagate and update optimizer learning rate
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.max_grad_norm)
            opt.step()

            # checkpoint model periodically
            if iterations % cfg.save_every == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "iterations": iterations,
                    "epoch": epoch,
                    "best_dev_acc": best_dev_acc,
                    "optimizer": opt.state_dict()
                }
                save_checkpoint(ckpt, cfg.save_path, iterations,
                                delete_old=True)

            # print progress message
            if iterations % cfg.print_every == 0:
                writer.add_scalar('train/loss', loss.item(), iterations)
                writer.add_scalar('train/acc', train_acc, iterations)
                for k, v in optional.items():
                    writer.add_scalar('train/' + k, v, iterations)

                opt_s = make_kv_string(optional)
                elapsed = int(time.time() - start)
                print("{:02d}:{:02d}:{:02d} epoch {:03d} "
                      "iter {:08d} loss {:.4f} {}".format(
                        elapsed // 3600, elapsed % 3600 // 60, elapsed % 60,
                        epoch, iterations, loss.item(), opt_s))

            # evaluate performance on validation set periodically
            if iterations % cfg.eval_every == 0:

                # switch model to evaluation mode
                model.eval()
                dev_iter.init_epoch()
                test_iter.init_epoch()

                # calculate accuracy on validation set
                dev_eval = evaluate(model, model.criterion, dev_iter)
                for k, v in dev_eval.items():
                    writer.add_scalar('dev/%s' % k, v, iterations)

                dev_eval_str = make_kv_string(dev_eval)
                print("# Evaluation dev : epoch {:2d} iter {:08d} {}".format(
                    epoch, iterations, dev_eval_str))

                # calculate accuracy on test set
                test_eval = evaluate(model, model.criterion, test_iter)
                for k, v in test_eval.items():
                    writer.add_scalar('test/%s' % k, v, iterations)

                test_eval_str = make_kv_string(test_eval)
                print("# Evaluation test: epoch {:2d} iter {:08d} {}".format(
                    epoch, iterations, test_eval_str))

                # update learning rate scheduler
                if isinstance(scheduler, ExponentialLR):
                    scheduler.step()
                else:
                    scheduler.step(dev_eval["acc"])

                # update best validation set accuracy
                if dev_eval["acc"] > best_dev_acc:

                    for k, v in dev_eval.items():
                        writer.add_scalar('best/dev/%s' % k, v, iterations)

                    for k, v in test_eval.items():
                        writer.add_scalar('best/test/%s' % k, v, iterations)

                    print("# New highscore {} iter {}".format(
                        dev_eval["acc"], iterations))

                    # print examples for highscore
                    dev_iter.init_epoch()
                    print_examples(model, dev_iter, input_field.vocab,
                                   label_field.vocab, cfg.save_path,
                                   iterations, n=5, writer=writer)

                    # found a model with better validation set accuracy
                    best_dev_acc = dev_eval["acc"]

                    # save model, delete previous 'best_*' files
                    ckpt = {
                        "model": model.state_dict(),
                        "cfg": cfg,
                        "iterations": iterations,
                        "epoch": epoch,
                        "best_dev_acc": best_dev_acc,
                        "best_test_acc": test_eval["acc"],
                        "optimizer": opt.state_dict()
                    }
                    save_checkpoint(
                        ckpt, cfg.save_path, iterations, prefix="best_ckpt",
                        dev_acc=dev_eval["acc"], test_acc=test_eval["acc"],
                        delete_old=True)

                if opt.param_groups[0]["lr"] < cfg.stop_lr_threshold:
                    print("Learning rate too low, stopping")
                    writer.close()
                    exit()

    writer.close()


if __name__ == "__main__":
    train()
