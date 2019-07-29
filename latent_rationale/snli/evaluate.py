import torch
from latent_rationale.snli.util import get_z_counts, get_n_correct


def evaluate(model, criterion, data_iter):

    model.eval()
    n_eval_correct, n_eval_total, eval_loss = 0, 0, 0

    # kuma statistics
    p2h_0, p2h_c, p2h_1 = 0, 0, 0
    h2p_0, h2p_c, h2p_1 = 0, 0, 0

    with torch.no_grad():
        for eval_batch_idx, eval_batch in enumerate(data_iter):
            answer = model(eval_batch)

            n_eval_correct += int(get_n_correct(eval_batch, answer))
            n_eval_total += int(eval_batch.batch_size)
            eval_loss += criterion(answer, eval_batch.label).sum().item()

            # statistics on p2h attention
            if hasattr(model, "prem2hypo_att"):
                z0, zc, z1 = get_z_counts(model.prem2hypo_att,
                                          model.prem_mask, model.hypo_mask)
                p2h_0 += z0
                p2h_c += zc
                p2h_1 += z1

            # statistics on h2p attention
            if hasattr(model, "hypo2prem_att"):
                z0, zc, z1 = get_z_counts(model.hypo2prem_att,
                                          model.hypo_mask, model.prem_mask)
                h2p_0 += z0
                h2p_c += zc
                h2p_1 += z1

            # statistics on p2h attention
            if hasattr(model, "prem2hypo_att"):
                z0, zc, z1 = get_z_counts(model.prem2hypo_att,
                                          model.prem_mask, model.hypo_mask)
                p2h_0 += z0
                p2h_c += zc
                p2h_1 += z1

            # statistics on h2p attention
            if hasattr(model, "hypo2prem_att"):
                z0, zc, z1 = get_z_counts(model.hypo2prem_att,
                                          model.hypo_mask, model.prem_mask)
                h2p_0 += z0
                h2p_c += zc
                h2p_1 += z1

    acc = 100. * n_eval_correct / n_eval_total

    result = dict(
        n_eval_correct=n_eval_correct, n_eval_total=n_eval_total,
        acc=acc, loss=eval_loss)

    if hasattr(model, "hypo2prem_att"):
        total = h2p_0 + h2p_c + h2p_1
        result["h2p_0"] = h2p_0 / total
        result["h2p_c"] = h2p_c / total
        result["h2p_1"] = h2p_1 / total
        result["h2p_selected"] = 1 - h2p_0 / total

    if hasattr(model, "prem2hypo_att"):
        total = p2h_0 + p2h_c + p2h_1
        result["p2h_0"] = p2h_0 / total
        result["p2h_c"] = p2h_c / total
        result["p2h_1"] = p2h_1 / total
        result["p2h_selected"] = 1 - p2h_0 / total

    return result
