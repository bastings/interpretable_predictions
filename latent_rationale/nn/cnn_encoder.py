from torch import nn


class CNNEncoder(nn.Module):
    """
    Returns a bag-of-words for a sequence of word embeddings.
    Ignores masked-out positions.
    """

    def __init__(self,
                 embedding_size: int = 300,
                 hidden_size: int = 200,
                 kernel_size: int = 5):
        super(CNNEncoder, self).__init__()
        padding = kernel_size // 2
        self.cnn = nn.Conv1d(embedding_size, hidden_size, kernel_size,
                             padding=padding, bias=True)

    def forward(self, x, mask, lengths):
        """

        :param x: sequence of word embeddings, shape [B, T, E]
        :param mask: byte mask that is 0 for invalid positions, shape [B, T]
        :param lengths: the lengths of each input sequence [B]
        :return:
        """
        # Conv1d Input:  (N, embedding_size E, T)
        # Conv1d Output: (N, hidden_size D,    T)
        x = x.transpose(1, 2)  # make [B, E, T]

        x = self.cnn(x)

        x = x.transpose(1, 2)  # make [B, T, D]
        x = x * mask.unsqueeze(-1).float()  # mask out padding
        x = x.sum(1) / lengths.unsqueeze(-1).float()  # normalize by sent length

        return None, x
