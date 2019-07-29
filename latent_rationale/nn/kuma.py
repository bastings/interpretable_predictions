import torch
from torch.nn import functional as F
import math


EPS = 1e-6


def convert_to_tensor(x):
    if isinstance(x, float):
        return torch.Tensor([x])
    return x


def lbeta(a, b):
    return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)


def lbeta(x):
    log_prod_gamma_x = torch.lgamma(x).sum(-1)
    log_gamma_sum_x = torch.lgamma(x.sum(-1))
    return log_prod_gamma_x - log_gamma_sum_x


def _harmonic_number(x):
    """
    From Tensorflow Probability.
    Compute the harmonic number from its analytic continuation.
    Derivation from [here](
    https://en.wikipedia.org/wiki/Digamma_function#Relation_to_harmonic_numbers)
    and [Euler's constant](
    https://en.wikipedia.org/wiki/Euler%E2%80%93Mascheroni_constant).
    Args:
      x: input float.
    Returns:
      z: The analytic continuation of the harmonic number for the input.
    """
    one = x.new_ones([1])
    return torch.digamma(x + one) - torch.digamma(one)


class RV:

    def params(self):
        raise NotImplementedError('Implement me')

    def sample(self, size=None):
        raise NotImplementedError('Implement me')

    def log_pdf(self, x):
        raise NotImplementedError('Implement me')

    def log_cdf(self, x):
        raise NotImplementedError('Implement me')

    def entropy(self):
        raise NotImplementedError('Implement me')

    def pdf(self, x):

        if isinstance(x, float):
            x = self.params()[0].new_tensor([x])

        return self.log_pdf(x).exp()

    def cdf(self, x):

        if isinstance(x, float):
            x = self.params()[0].new_tensor([x])

        return self.log_cdf(x).exp()


class RelaxedBinary(RV):
    """This is a variable such that values live in the open interval (0, 1)"""
    pass


class Kuma(RelaxedBinary):
    """
    A Kumaraswamy, or Kuma for short, is similar to a Beta distribution, though not an exponential family.
     Kuma variables are specified by two shape parameters, similar to Beta, though for settings that typically
     yield a symmetric Beta won't necessarily yield a symmetric Kuma.

    X ~ Kuma(a,b)
        where a, b > 0

    Or equivalently,
        U ~ U(0,1)
        x = (1 - (1 - u)^(1/b))^(1/a)

    In practice we sample from U(0 + eps, 1 - eps) for some small positive constant eps to avoid instabilities.
    """

    def __init__(self, params: list):
        self.a = params[0]
        self.b = params[1]

    def params(self):
        return [self.a, self.b]

    def mean(self):
        return kuma_moments(self.a, self.b, 1)

    def sample(self, size=None, eps=0.001):

        if size is None:
            u = torch.zeros_like(self.a).uniform_(eps, 1. - eps)
        elif isinstance(size, int):
            u = self.a.new_zeros([size] + list(self.a.shape)).uniform_(eps, 1. - eps)
        else:  # assume full shape
            u = self.a.new_zeros(size).uniform_(eps, 1. - eps)

        return (1. - (1 - u) ** self.b.reciprocal()) ** self.a.reciprocal()

    def log_pdf(self, x):
        """
        Kuma(x|a, b) = U(s(x)|0, 1) |det J_s|
            where x = t(u) and u = s(x) and J_s is the Jacobian matrix of s(x)
        """
        if isinstance(x, float):
            x = self.params()[0].new_tensor([x])

        t1 = torch.log(self.a) + torch.log(self.b)
        t2 = (self.a - 1. + EPS) * torch.log(x)
        pow_x_a = (x ** self.a) + EPS
        t3b = torch.log(1. - pow_x_a)
        t3 = (self.b - 1. + EPS) * t3b
        return t1 + t2 + t3

    def log_cdf(self, x):
        if isinstance(x, float):
            x = self.params()[0].new_tensor([x])

        r = 1. - ((1. - (x ** self.a)) ** self.b)
        r = torch.log(r + EPS)
        return r.clamp(math.log(EPS), math.log(1-EPS))


class StretchedVariable(RelaxedBinary):
    """
    A continuous variable over the open interval [left, right].

    X ~ StretchedVariable(RelaxedBinary, [left, right])
            left < 0 and right > 1

        Or equivalently,

            Y ~ RelaxedBinary()
            x = location + y * scale
                where location = left
                and scale = right - left
    """

    def __init__(self, dist: RelaxedBinary, support: list):
        """
        :param dist: a RelaxedBinary variable (e.g. BinaryConcrete or Kuma)
        :param support: a pair specifying the limits of the stretched support (e.g. [-1, 2])
            we use these values to compute location = pair[0] and scale = pair[1] - pair[0]
        """
        assert isinstance(dist,
                          RelaxedBinary), 'I need a RelaxedBinary variable, got %s' % type(
            dist)
        assert support[0] < support[
            1], 'I need an ordered support, got %s' % support
        self._dist = dist
        self.loc = support[0]
        self.scale = support[1] - support[0]

    def params(self):
        return self._dist.params()

    def sample(self, size=None):
        # sample a relaxed binary variable
        x_ = self._dist.sample(size=size)
        # and stretch it
        return x_ * self.scale + self.loc

    def log_pdf(self, x):

        if isinstance(x, float):
            x = self.params()[0].new_tensor([x])

        if isinstance(self.scale, float):
            self.scale = self.params()[0].new_tensor([self.scale])

        # shrink the stretched variable
        x_ = (x - self.loc) / self.scale
        # and assess the stretched pdf using the original pdf
        # see eq 25 (left) of Louizos et al
        return self._dist.log_pdf(x_) - torch.log(self.scale)

    def log_cdf(self, x):

        if isinstance(x, float):
            x = self.params()[0].new_tensor([x])

        # shrink the stretched variable
        x_ = (x - self.loc) / self.scale
        # assess its cdf
        # see eq 25 (right) of Louizos et al
        return self._dist.log_cdf(x_).clamp(math.log(EPS), math.log(1-EPS))


class HardBinary(RV):
    """
    A continuous variable over the closed interval [0, 1] which can assign non-zero probability mass
    to {0} and {1} (which are sets of zero measure in a standard RelaxedBinary or StretchedVariable).

    X ~ HardBinary(StretchedVariable)

    Or equivalently,

        Y ~ StretchedVariable()
        x = hardsigmoid(y)
    """

    def __init__(self, dist: StretchedVariable):
        assert isinstance(dist, StretchedVariable), \
            "I need a stretched variable"
        self._dist = dist

    def params(self):
        return self._dist.params()

    def sample(self, size=None):
        # sample a stretched variable and rectify it
        x_ = self._dist.sample(size=size)
        return F.hardtanh(x_, min_val=0., max_val=1.)

    def log_pdf(self, x):
        """
        We obtain pdf(0) by integrating the stretched variable over the interval [left, 0]
            HardBinary.pdf(0) = StretchedVariable.cdf(0)
         and pdf(1) by integrating the stretched variable over the interval [1, right], or equivalently,
            HardBinary.pdf(1) = 1 - StretchedVariable.cdf(1)
         finally, for values in the open (0, 1) we scale the pdf of the stretched variable by the remaining probability mass
            HardBinary.pdf(x) = StretchedVariable.pdf(x) * (1 - HardBinary.pdf(0) - HardBinary.pdf(1))

        See that the total mass over the discrete set {0, 1} is
            HardBinary.pdf(0) + HardBinary.pdf(1)
         in other words, with this probability we will be sampling a discrete value.
         Whenever this probability is greater than 0.5, most probability mass is away from continuous samples.
        """

        if isinstance(x, float):
            x = self.params()[0].new_tensor([x])

        # cache these for faster computation
        log_cdf_0 = self._dist.log_cdf(x.new_zeros(1))
        cdf_1 = self._dist.cdf(x.new_ones(1))

        # first we fix log_pdf for 0s and 1s
        log_p = torch.where(
            x == 0.,
            log_cdf_0,             # log Q(0)
            torch.log(1. - cdf_1)  # log (1-Q(1))
        )
        # then for those that are in the open (0, 1)
        log_p = torch.where(
            (0. < x) & (x < 1.), self._dist.log_pdf(x), log_p)
        # see eq 26 of Louizos et al
        return log_p

    def log_cdf(self, x):
        """
        Note that HardKuma.cdf(0) = HardKuma.pdf(0) by definition of HardKuma.pdf(0),
         also note that HardKuma.cdf(1) = 1 by definition because
         the support of HardKuma is the *closed* interval [0, 1]
         and not the open interval (left, right) which is the support of the stretched variable.
        """
        if isinstance(x, float):
            x = self.params()[0].new_tensor([x])

        log_c = torch.where(
            x < 1.,
            self._dist.log_cdf(x),
            x.new_zeros(x.size())  # all of the mass
        )
        return log_c.clamp(math.log(EPS), math.log(1-EPS))


class HardKuma(HardBinary):

    def __init__(self, params: list, support: list):
        super(HardKuma, self).__init__(StretchedVariable(Kuma(params), support))

        # shortcut to underlying a and b
        self.a = self._dist._dist.a
        self.b = self._dist._dist.b

    def mean(self):
        return kuma_moments(self.a, self.b, 1)


def kuma_mean(a, b):
    return kuma_moments(a, b, 1)


def kuma_moments(a, b, n):
    """
    Computes nth moment of Kumaraswamy using using torch.lgamma
    :param a:
    :param b:
    :param n:
    :return: nth moment
    """
    arg1 = 1 + n / a
    log_value = torch.lgamma(arg1) + torch.lgamma(b) - torch.lgamma(arg1 + b)
    return b * torch.exp(log_value)
