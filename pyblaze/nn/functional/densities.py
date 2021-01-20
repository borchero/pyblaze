import math
import torch

def log_prob_standard_normal(x):
    """
    Computes the log-probability of observing the given data under a (multivariate) standard Normal
    distribution. Although this function is equivalent to the :code:`log_prob` method of the
    :class:`torch.distributions.MultivariateNormal` class, this implementation is much more
    efficient due to the restriction to standard Normal.

    Parameters
    ----------
    x: torch.Tensor [N, D]
        The samples whose log-probability shall be computed (number of samples N, dimensionality D).

    Returns
    -------
    torch.Tensor [N]
        The log-probabilities for all samples.
    """
    dim = x.size(1)
    const = dim * math.log(2 * math.pi)
    norm = torch.einsum('ij,ij->i', x, x)
    return -0.5 * (const + norm)


def log_prob_standard_gmm(x, means):
    """
    Computes the log-probability of observing the given data under a GMM consisting of
    (multivariate) standard normal distributions. Each component is assigned the same weight.

    Parameters
    ----------
    x: torch.Tensor [N, D]
        The samples whose log-probability shall be computed (number of samples N,
        dimensionality D).
    means: torch.Tensor [M, D]
        The means of the GMM.

    Returns
    -------
    torch.Tensor [N]
        The log-probabilities for all samples.
    """
    num_datapoints, dim = x.size()
    num_components = means.size(0)

    const = dim * math.log(2 * math.pi)
    xx = torch.einsum('ij,ij->i', x, x).view(num_datapoints, 1)
    mm = torch.einsum('ij,ij->i', means, means).view(1, num_components)
    xm = x.matmul(means.t())
    log_probs = -0.5 * (const + xx - 2 * xm + mm)

    return torch.logsumexp(log_probs - math.log(num_components), dim=1)


def generate_random_gmm(num_components, dim, seed=None):
    """
    Generates the means of a GMM with the specified number of components. Each mean is sampled from
    a standard normal distribution of the given dimension.

    Parameters
    ----------
    num_components: int
        The number of components in the GMM.
    dim: int
        The dimensionality of the GMM.
    seed: int, default: None
        The seed to use for randomly sampling the components.

    Returns
    -------
    torch.Tensor [N, D]
        The means of the GMM (number of components N, dimensionality D).
    """
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    means = torch.randn(num_components, dim, generator=generator)
    return means
