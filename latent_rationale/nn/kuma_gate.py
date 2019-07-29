# coding: utf-8
import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, Dropout, Softplus, Tanh, ReLU

from latent_rationale.nn.kuma import Kuma, HardKuma

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class KumaGate(nn.Module):
    """
    Computes a Hard Kumaraswamy Gate
    """

    def __init__(self, in_features, out_features=1, support=(-0.1, 1.1),
                 dist_type="hardkuma"):
        super(KumaGate, self).__init__()

        self.dist_type = dist_type

        self.layer_a = Sequential(
            Linear(in_features, out_features),
            Softplus()
        )
        self.layer_b = Sequential(
            Linear(in_features, out_features),
            Softplus()
        )

        # support must be Tensors
        s_min = torch.Tensor([support[0]]).to(device)
        s_max = torch.Tensor([support[1]]).to(device)
        self.support = [s_min, s_max]

        self.a = None
        self.b = None

    def forward(self, x, mask=None):
        """
        Compute latent gate
        :param x: word represenatations [B, T, D]
        :param mask: [B, T]
        :return: gate distribution
        """

        a = self.layer_a(x)
        b = self.layer_b(x)

        a = a.clamp(1e-6, 100.)  # extreme values could result in NaNs
        b = b.clamp(1e-6, 100.)  # extreme values could result in NaNs

        self.a = a
        self.b = b

        # we return a distribution (from which we can sample if we want)
        if self.dist_type == "kuma":
            dist = Kuma([a, b])

        elif self.dist_type == "hardkuma":
            dist = HardKuma([a, b], support=self.support)
        else:
            raise ValueError("unknown dist")

        return dist
