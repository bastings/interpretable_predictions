from torch import nn


class BOWEncoder(nn.Module):
    """
    Returns a bag-of-words for a sequence of word embeddings.
    Ignores masked-out positions.
    """

    def __init__(self):
        super(BOWEncoder, self).__init__()

    def forward(self, x, mask, lengths):
        """

        :param x: sequence of word embeddings, shape [B, T, E]
        :param mask: byte mask that is 0 for invalid positions, shape [B, T]
        :param lengths: the lengths of each input sequence [B]
        :return:
        """
        bow = x * mask.unsqueeze(-1).float()
        bow = bow.sum(1)                           # sum over time to get [B, E]
        bow = bow / lengths.unsqueeze(-1).float()  # normalize by sent length
        return None, bow
