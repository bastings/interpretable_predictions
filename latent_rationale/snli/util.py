import os
from argparse import ArgumentParser
import torch
from hashlib import md5
import numpy as np
import glob
import re
from torch.nn import functional as F

from latent_rationale.snli.constants import UNK_TOKEN, PAD_TOKEN, INIT_TOKEN
from latent_rationale.snli.plotting import plot_heatmap
from latent_rationale.snli.text import data


BIN_REGEX = re.compile(r"\( | \)")
NON_BIN_REGEX = re.compile(r"\([A-Z.,:$]+|\)")


def masked_softmax(t, mask, dim=-1):
    t = torch.where(mask, t, t.new_full([1], float('-inf')))
    return F.softmax(t, dim=dim)


def get_data_fields(glove_words, lowercase=False,
                    init_token=INIT_TOKEN):

    not_in_glove = set()

    def _tokens_from_binary_parse(s):
        tokens = re.sub(BIN_REGEX, "", s).split()
        tokens = unk_unknown_tokens(
            tokens, glove_words=glove_words, not_in_glove=not_in_glove)
        return tokens

    def _tokens_from_non_binary_parse(s):
        tokens = re.sub(NON_BIN_REGEX, "", s).split()
        tokens = unk_unknown_tokens(
            tokens, glove_words=glove_words, not_in_glove=not_in_glove)
        return tokens

    input_field = data.Field(
        lower=lowercase, tokenize=_tokens_from_binary_parse,
        batch_first=True, include_lengths=True,
        init_token=init_token, pad_token=PAD_TOKEN, unk_token=UNK_TOKEN)

    label_field = data.Field(
        sequential=False, batch_first=True, unk_token=None)

    return input_field, label_field, not_in_glove


def unk_unknown_tokens(tokens, n=100, lowercase=False,
                       glove_words=None, not_in_glove=None):
    """
    Hash unknown words into N different UNK-classes

    :param tokens:
    :param n: hash tokens into this many classes
    :param lowercase:
    :param glove_words: a set with all valid glove words
    :param not_in_glove: an empty set where we store words that were not in glove
    :return:
    """
    new_tokens = []
    for token in tokens:

        if lowercase:
            token = token.lower()

        if token not in glove_words:
            not_in_glove.add(token)
            hash_idx = hash_token(token, n=n)
            token = "<unk_{:02d}>".format(hash_idx)

        new_tokens.append(token)
    return new_tokens


def get_n_correct(batch, answer):
    """get number of correct predictions (float)"""
    return (torch.max(answer, 1)[1].view(
        batch.label.size()) == batch.label).float().sum().item()


def find_ckpt_in_directory(path):
    for f in os.listdir(os.path.join(path, "")):
        if f.startswith('best_ckpt'):
            return os.path.join(path, f)


def save_checkpoint(ckpt, save_path, iterations, prefix="ckpt",
                    dev_acc=None, test_acc=None, delete_old=False):

    ckpt_prefix = os.path.join(save_path, prefix)
    ckpt_path = ckpt_prefix + "_iter_{:08d}".format(iterations)

    if dev_acc is not None:
        ckpt_path += "_devacc_{:4.2f}".format(dev_acc)

    if test_acc is not None:
        ckpt_path += "_testacc_{:4.2f}".format(test_acc)

    ckpt_path += ".pt"

    try:
        torch.save(ckpt, ckpt_path)
    except IOError:
        print("Error while saving checkpoint (iteration %d)" % iterations)

    if delete_old:
        try:
            for f in glob.glob(ckpt_prefix + '*'):
                if f != ckpt_path:
                    os.remove(f)
        except IOError:
            print("Error while deleting old checkpoint")


def load_glove_words(word_vectors):
    print("Loading Glove dictionary: {}".format(word_vectors))
    words = set()
    path = os.path.join("data/snli", word_vectors + ".words.txt")
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            word = line.rstrip()
            words.add(word)
    print("Loaded:", len(words), "words")
    return words


def get_z_counts(att, prem_mask, hypo_mask):
    """
    Compute z counts (number of 0, continious, 1 elements).
    :param att: similarity matrix [B, prem, hypo]
    :param prem_mask:
    :param hypo_mask:
    :return: z0, zc, z1
    """
    # mask out all invalid positions with -1
    att = torch.where(hypo_mask.unsqueeze(1), att, att.new_full([1], -1.))
    att = torch.where(prem_mask.unsqueeze(2), att, att.new_full([1], -1.))

    z0 = (att == 0.).sum().item()
    zc = ((0 < att) & (att < 1)).sum().item()
    z1 = (att == 1.).sum().item()

    assert (att > -1).sum().item() == z0 + zc + z1, "mismatch"

    return z0, zc, z1


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def print_config(config):
    for k, v in vars(config).items():
        print("%22s : %16s" % (k, str(v)))
    print()


def print_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    print("Total params: %d" % n_params)
    for name, p in model.named_parameters():
        if p.requires_grad:
            print("%30s : %12s" % (name, list(p.size())))
        else:
            print("%30s : %12s (no-grad)" % (name, list(p.size())))
    print()


def hash_token(token, n=100):
    return int(md5(token.encode()).hexdigest(), 16) % n


def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    import os, errno

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def remove_padding(text, pad_token):
    try:
        cut = text.index(PAD_TOKEN)
        text = text[:cut]
    except ValueError:  # no padding present
        pass

    return text


def extract_attention(model, data_iter, input_vocab, answer_vocab):
    """
    :param model:
    :param data_iter:
    :param input_vocab:
    :param answer_vocab:
    :return:
    """

    if not hasattr(model, "prem2hypo_att"):
        return

    data_iter.init_epoch()
    model.eval()

    p2h_att = []
    h2p_att = []
    prems = []
    hypos = []
    predictions = []
    targets = []

    with torch.no_grad():
        for i, batch in enumerate(data_iter, 1):

            result = model(batch)

            for j in range(batch.batch_size):

                prem = [input_vocab.itos[x] for x in batch.premise[0][j]]
                hypo = [input_vocab.itos[x] for x in batch.hypothesis[0][j]]

                prem = remove_padding(prem, PAD_TOKEN)
                hypo = remove_padding(hypo, PAD_TOKEN)

                label = answer_vocab.itos[batch.label[j]]
                answer = answer_vocab.itos[result.argmax(dim=-1)[j]]

                prem2hypo_att = model.prem2hypo_att[j].cpu().numpy()
                hypo2prem_att = model.hypo2prem_att[j].cpu().numpy()

                prem2hypo_att = prem2hypo_att[:len(prem), :len(hypo)]
                hypo2prem_att = hypo2prem_att[:len(hypo), :len(prem)]

                targets.append(label)
                predictions.append(answer)
                p2h_att.append(prem2hypo_att)
                h2p_att.append(hypo2prem_att)
                prems.append(prem)
                hypos.append(hypo)

    return p2h_att, h2p_att, prems, hypos, predictions, targets


def print_examples(model, data_iter, input_vocab, answer_vocab, save_path,
                   iterations, n=3, writer=None, skip_null=True):
    """

    :param model:
    :param data_iter:
    :param input_vocab:
    :param answer_vocab:
    :param save_path:
    :param iterations:
    :param n:
    :param writer: Tensorboard writer to write attention images to Tensorboard
    :param skip_null: do not show NULL (first) symbol in plot
    :return:
    """
    data_iter.init_epoch()
    model.eval()
    n_printed = 0

    with torch.no_grad():
        for i, batch in enumerate(data_iter, 1):

            result = model(batch)

            for j in range(batch.batch_size):

                prem = [input_vocab.itos[x] for x in batch.premise[0][j]]
                hypo = [input_vocab.itos[x] for x in batch.hypothesis[0][j]]

                try:
                    cut = prem.index(PAD_TOKEN)
                    prem = prem[:cut]
                except ValueError:
                    pass

                try:
                    cut = hypo.index(PAD_TOKEN)
                    hypo = hypo[:cut]
                except ValueError:
                    pass

                label = answer_vocab.itos[batch.label[j]]
                answer = answer_vocab.itos[result.argmax(dim=-1)[j]]

                # extract attention matrices
                if hasattr(model, "prem2hypo_att"):
                    prem2hypo_att = model.prem2hypo_att[j].cpu().numpy()
                    hypo2prem_att = model.hypo2prem_att[j].cpu().numpy()

                    if skip_null:
                        prem2hypo_att = prem2hypo_att[1:, 1:]
                        hypo2prem_att = hypo2prem_att[1:, 1:]
                        prem = prem[1:]
                        hypo = hypo[1:]

                    # attention is normalized by last dimension, so columns here
                    name = "ex{:02d}_prem2hypo_att".format(n_printed)
                    if writer is not None:
                        writer.add_image("data/" + name,
                                         prem2hypo_att[None, :, :],
                                         iterations)
                    path = os.path.join(save_path, name + ".pdf")
                    plot_heatmap(prem2hypo_att, row_labels=prem,
                                 column_labels=hypo, output_path=path)

                    # attention is normalized by last dimension, so columns here
                    name = "ex{:02d}_hypo2prem_att".format(n_printed)
                    if writer is not None:
                        writer.add_image("data/" + name,
                                         hypo2prem_att[None, :, :],
                                         iterations)
                    path = os.path.join(save_path, name + ".pdf")
                    plot_heatmap(hypo2prem_att, row_labels=prem,
                                 column_labels=hypo, output_path=path)

                # extract multi-head self-attention matrices
                if hasattr(model, "prem_self_att_samples"):
                    for k, a in enumerate(model.prem_self_att_samples):
                        prem_self_att = a[j].cpu().numpy()
                        prem_self_att = prem_self_att[:len(prem), :len(prem)]
                        name = "ex{:02d}_prem_sa{}".format(n_printed, k)
                        if writer is not None:
                            writer.add_image("data/" + name,
                                             prem_self_att[None, :, :],
                                             iterations)
                        name = name + ".pdf"
                        plot_heatmap(prem_self_att,
                                     row_labels=prem, column_labels=prem,
                                     output_path=os.path.join(save_path, name))

                if hasattr(model, "hypo_self_att_samples"):
                    for k, a in enumerate(model.hypo_self_att_samples):
                        hypo_self_att = a[j].cpu().numpy()
                        hypo_self_att = hypo_self_att[:len(hypo), :len(hypo)]
                        name = "ex{:02d}_hypo_sa{}".format(n_printed, k)
                        if writer is not None:
                            writer.add_image("data/" + name,
                                             hypo_self_att[None, :, :],
                                             iterations)
                        name = name + ".pdf"
                        plot_heatmap(hypo_self_att,
                                     row_labels=prem, column_labels=prem,
                                     output_path=os.path.join(save_path, name))

                # extract self-attention matrices
                if hasattr(model, "prem_self_att") and \
                        model.prem_self_att is not None:
                    prem_self_att = model.prem_self_att[j].cpu().numpy()
                    hypo_self_att = model.hypo_self_att[j].cpu().numpy()

                    plot_heatmap(prem_self_att,
                                 row_labels=prem, column_labels=prem,
                                 output_path=os.path.join(
                                     save_path,
                                     "ex%02d_prem_self_att.pdf" % n_printed))

                    plot_heatmap(hypo_self_att,
                                 row_labels=hypo, column_labels=hypo,
                                 output_path=os.path.join(
                                     save_path,
                                     "ex%02d_hypo_self_att.pdf" % n_printed))

                print("Example {}".format(n_printed))
                print("{:11} : {}".format("Premise:", " ".join(prem)))
                print("{:11} : {}".format("Hypothesis:", " ".join(hypo)))
                print("{:11} : {}".format("Label:", label))
                print("{:11} : {}".format("Prediction:", answer))
                print()

                n_printed += 1

                if n_printed == n:
                    return


def get_predict_args():
    parser = ArgumentParser(description='PyTorch/torchtext SNLI example')
    parser.add_argument('--ckpt', type=str, default="path_to_checkpoint")
    args = parser.parse_args()
    return args


def get_args():
    parser = ArgumentParser(description='SNLI')

    parser.add_argument('--save_path', type=str, default='results/snli/default')
    parser.add_argument('--resume_snapshot', type=str, default='')

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--proj_size', type=int, default=200)
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--n_layers', type=int, default=1)

    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=1000)

    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--min_lr', type=float, default=5e-5)
    parser.add_argument('--lr_decay', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--patience', type=int, default=10000)
    parser.add_argument('--max_grad_norm', type=float, default=5.)
    parser.add_argument('--stop_lr_threshold', type=float, default=1e-5)

    parser.add_argument('--max_relative_distance', type=int, default=11)
    parser.add_argument('--model',
                        choices=["recurrent", "decomposable"],
                        default="decomposable")
    parser.add_argument('--dist', choices=["", "hardkuma"],
                        default="")
    parser.add_argument('--self-attention', action='store_true',
                        help="intra-sentence attention (Decomposable model)")
    parser.add_argument('--no-bidirectional', action='store_false', dest='birnn')

    # control Hard Kuma sparsity
    parser.add_argument('--selection', type=float, default=1.0)

    # lagrange settings
    parser.add_argument('--lagrange_lr', type=float, default=0.01,
                        help="learning rate for lagrange")
    parser.add_argument('--lagrange_alpha', type=float, default=0.99,
                        help="alpha for computing the running average")
    parser.add_argument('--lambda_init', type=float, default=1e-5,
                        help="initial value for lambda")

    # misc
    parser.add_argument('--no-projection', action='store_false',
                        dest='projection')
    parser.add_argument('--mask-diagonal', action='store_true')
    parser.add_argument('--overwrite', action='store_true',
                        help="erase save_path if it exists")
    parser.add_argument('--no-emb-normalization', action='store_false',
                        dest='normalize_embeddings')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    parser.add_argument('--word_vectors', type=str,
                        default='glove.840B.300d')

    args = parser.parse_args()
    return args
