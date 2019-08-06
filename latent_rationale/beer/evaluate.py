import numpy as np
import torch
from latent_rationale.beer.util import prepare_minibatch, get_minibatch, \
    decorate_token

from collections import defaultdict


def dump_rationales(model, data, path, device=None, batch_size=256):

    if not hasattr(model, "z"):
        return

    model.eval()  # disable dropout
    sent_id = 0

    with open(path, mode="w", encoding="utf-8") as f:

        for mb in get_minibatch(data, batch_size=batch_size, shuffle=False):
            x, targets, reverse_map = prepare_minibatch(
                mb, model.vocab, device=device, sort=True)

            with torch.no_grad():
                logits = model(x)

                if hasattr(model, "alphas"):
                    alphas = model.alphas
                else:
                    alphas = None

                if hasattr(model, "z"):
                    z = model.z
                    bsz, num_samples, max_time, _ = z.size()
                    z = z.view(bsz, num_samples, max_time)
                else:
                    z = None

            # the inputs were sorted to enable packed_sequence for LSTM
            # we need to reverse sort them so that they correspond
            # to the original order

            # reverse sort
            alphas = alphas[reverse_map] if alphas is not None else None
            z = z[reverse_map] if z is not None else None  # [B,T]

            for i, ex in enumerate(mb):
                tokens = ex.tokens

                if alphas is not None:
                    alpha = alphas[i][:len(tokens)]
                    alpha = alpha[None, :]

                # z is [batch_size, num_samples, time]
                if z is not None:

                    z_samples = z[i, :, :len(tokens)]

                    for sample_id in range(z_samples.shape[0]):

                        z_sample = z_samples[sample_id]
                        example = []

                        for ti, zi in zip(tokens, z_sample):
                            example.append(decorate_token(ti, zi))

                        f.write(" ".join(example))
                        f.write("\n")

                sent_id += 1


def evaluate_rationale(model, data, aspect=None, batch_size=256, device=None,
                       path=None):
    """Precision on annotated rationales.

    This works in a simple way:
    We have a predicted vector z
    We have a gold annotation  z_gold
    We take a logical and (to intersect the two)
    We sum the number of words in the intersection and divide by the total
    number of selected words.
    """

    assert aspect is not None, "provide aspect"
    assert device is not None, "provide device"

    if not hasattr(model, "z"):
        return

    if path is not None:
        ft = open(path, mode="w", encoding="utf-8")
        fz = open(path + ".z", mode="w", encoding="utf-8")

    model.eval()  # disable dropout
    sent_id, correct, total, macro_prec_total, macro_n = 0, 0, 0, 0, 0

    for mb in get_minibatch(data, batch_size=batch_size, shuffle=False):
        x, targets, reverse_map = prepare_minibatch(
            mb, model.vocab, device=device, sort=True)

        with torch.no_grad():
            logits = model(x)

            # attention alphas
            if hasattr(model, "alphas"):
                alphas = model.alphas
            else:
                alphas = None

            # rationale z
            if hasattr(model, "z"):
                z = model.z  # [B, T]
                bsz, max_time = z.size()
            else:
                z = None

        # the inputs were sorted to enable packed_sequence for LSTM
        # we need to reverse sort them so that they correspond
        # to the original order

        # reverse sort
        alphas = alphas[reverse_map] if alphas is not None else None
        z = z[reverse_map] if z is not None else None  # [B,T]

        # evaluate each sentence in this minibatch
        for mb_i, ex in enumerate(mb):
            tokens = ex.tokens
            annotations = ex.annotations

            # assuming here that annotations only has valid ranges for the
            # current aspect
            if aspect > -1:
                assert len(annotations) == 1, "expected only 1 aspect"

            if alphas is not None:
                alpha = alphas[mb_i][:len(tokens)]
                alpha = alpha[None, :]

            # z is [batch_size, time]
            if z is not None:

                z_ex = z[mb_i, :len(tokens)]  # i for minibatch example
                z_ex_nonzero = (z_ex > 0).float()
                z_ex_nonzero_sum = z_ex_nonzero.sum().item()

                # list of decorated tokens for this single example, to print
                example = []
                for ti, zi in zip(tokens, z_ex):
                    example.append(decorate_token(ti, zi))

                # write this sentence
                ft.write(" ".join(example))
                ft.write("\n")
                fz.write(" ".join(["%.4f" % zi for zi in z_ex]))
                fz.write("\n")

                # skip if no gold rationale for this sentence
                if aspect >= 0 and len(annotations[0]) == 0:
                    continue

                # compute number of matching tokens & precision
                matched = sum(1 for i, zi in enumerate(z_ex) if zi > 0 and
                              any(interval[0] <= i < interval[1]
                                  for a in annotations for interval in a))

                precision = matched / (z_ex_nonzero_sum + 1e-9)

                macro_prec_total += precision
                correct += matched
                total += z_ex_nonzero_sum
                if z_ex_nonzero_sum > 0:
                    macro_n += 1

                # print(matched, end="\t")

            sent_id += 1
    # print()
    # print("new correct", correct, "total", total)

    precision = correct / (total + 1e-9)
    macro_precision = macro_prec_total / (float(macro_n) + 1e-9)

    try:
        ft.close()
        fz.close()
    except IOError:
        print("Error closing file(s)")

    return precision, macro_precision


def get_examples(model, data, num_examples=3, batch_size=1, device=None):
    """Prints examples"""

    model.eval()  # disable dropout
    count = 0

    if not hasattr(model, "z"):
        return

    for mb in get_minibatch(data, batch_size=batch_size, shuffle=False):

        if count == num_examples:
            break

        x, targets, _ = prepare_minibatch(mb, model.vocab, device=device)
        with torch.no_grad():
            output = model(x)

            if hasattr(model, "z"):
                z = model.z.cpu().numpy().flatten()
                example = []
                for ti, zi in zip(mb[0].tokens, z):
                    example.append(decorate_token(ti, zi))
                # print("Example %d:" % count, " ".join(output))
                yield example
                count += 1


def evaluate_loss(model, data, batch_size=256, device=None, cfg=None):
    """
    Loss of a model on given data set (using minibatches)
    Also computes some statistics over z assignments.
    """
    model.eval()  # disable dropout
    total = defaultdict(float)
    total_examples = 0
    total_predictions = 0

    for mb in get_minibatch(data, batch_size=batch_size, shuffle=False):

        x, targets, _ = prepare_minibatch(mb, model.vocab, device=device)
        mask = (x != 1)

        batch_examples = targets.size(0)
        batch_predictions = np.prod(list(targets.size()))

        total_examples += batch_examples
        total_predictions += batch_predictions

        with torch.no_grad():
            output = model(x)
            loss, loss_opt = model.get_loss(output, targets, mask=mask)
            total["loss"] += loss.item() * batch_examples

            # e.g. mse_loss, loss_z_x, sparsity_loss, coherence_loss
            for k, v in loss_opt.items():
                total[k] += v * batch_examples

    result = {}
    for k, v in total.items():
        if not k.startswith("z_num"):
            result[k] = v / float(total_examples)

    if "z_num_1" in total:
        z_total = total["z_num_0"] + total["z_num_c"] + total["z_num_1"]
        selected = total["z_num_1"] / float(z_total)
        result["p1r"] = selected
        result["z_num_0"] = total["z_num_0"]
        result["z_num_c"] = total["z_num_c"]
        result["z_num_1"] = total["z_num_1"]

    return result
