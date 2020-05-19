import torch
import torch.distributions as D
from torch.utils.data import IterableDataset

# pylint: disable=abstract-method
class NoiseDataset(IterableDataset):
    """
    Infinite dataset for generating noise from a given probability distribution. Usually to be used
    with generative adversarial netwroks.
    """

    def __init__(self, distribution=D.Normal(torch.zeros(2), torch.ones(2))):
        """
        Initializes a new dataset where noise is sampled from the given distribution.

        Parameters
        ----------
        distribution: torch.distributions.Distribution
            The noise type to use.
        """
        super().__init__()
        self.distribution = distribution

    def __iter__(self):
        while True:
            yield self.distribution.sample()
