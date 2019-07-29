import torch


def get_relative_positions(size, max_dist, device=None):
    """
    Returns the IDs of relative distances, with a max. for bucketing
    e.g. for sentence with 3 words, the relative positions look like:

     0  1  2
    -1  0  1
    -2 -1  0

    to index distance embeddings, we add the maximum distance:
     0+max_dist 1+max_dist, 2+max_dist
    -1+max_dist .. etc.
    etc.

    values larger than max_dist or smallar than -max_dist are clipped

    :param size:
    :param max_dist: maximum relative distance
    :param device: device to create output tensor

    :return: indices for relative distances
    """
    with torch.no_grad():
        v = torch.arange(size, device=device)
        v1 = v.unsqueeze(0)  # broadcast over rows
        v2 = v.unsqueeze(1)  # broadcast over columns
        d = v1 - v2
        d = d.clamp(-max_dist, max_dist) + max_dist
        return d
