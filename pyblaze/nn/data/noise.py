import torch
import torch.distributions as D
from torch.utils.data import IterableDataset

# pylint: disable=abstract-method
class NoiseDataset(IterableDataset):
    """
    Infinite dataset for generating noise from a given probability distribution. Usually to be used
    with generative adversarial networks.
    """

    def __init__(self, latent_dim=2, distribution=None):
        """
        Initializes a new dataset where noise is sampled from the given distribution.
        If no distribution is given, noise is sampled from a multivariate normal distribution
        with a certain latent dimension.

        Parameters
        ----------
        latent_dim: int
            The latent dimension for the Normal Distribution the noise is sampled from.
        distribution: torch.distributions.Distribution
            The noise type to use. Overrides setting of latent_dim if specified.
        """
        super().__init__()
        if distribution is None:
            self.distribution = D.Normal(torch.zeros(latent_dim), torch.ones(latent_dim))
        else:
            self.distribution = distribution

    def __iter__(self):
        while True:
            yield self.distribution.sample()


class LabeledNoiseDataset(NoiseDataset):
    """
    Infinite dataset for generating noise from a given probability distribution. Usually to be used
    with generative adversarial networks conditioned on class labels.
    """

    def __init__(self, latent_dim=2, num_classes=10, distribution=None, categorical=None):
        """
        Initializes a new dataset where noise and a label is sampled from the given distribution.
        If no distribution is given, noise is sampled from a multivariate normal distribution
        with a certain latent dimension and the label is sampled from a categorical distribution.

        Parameters
        ----------
        latent_dim: int
            The latent dimension for the Normal Distribution the noise is sampled from.
        num_classes: int
            Number of classes for the Categorical Distribution the label is sampled from.
        distribution: torch.distributions.Distribution
            The noise type to use. Overrides setting of latent_dim if specified.
        categorical: torch.distributions.Distribution
            The distribution to sample labels from. Overrides setting of num_classes if specified.
        """
        super().__init__(latent_dim=latent_dim, distribution=distribution)

        if categorical is None:
            self.categorical = D.Categorical(torch.Tensor([1.0 / num_classes] * num_classes))
        else:
            self.categorical = categorical

    def __iter__(self):
        it = iter(super())
        while True:
            # pylint: disable=stop-iteration-return
            yield next(it), self.categorical.sample()
