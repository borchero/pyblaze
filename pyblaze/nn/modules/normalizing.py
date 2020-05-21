import math
import torch
import torch.nn as nn

class NormalizingFlow(nn.Module):
    """
    In general, a normalizing flow is a module to transform an initial density into another one
    (usually a more complex one) via a sequence of invertible transformations. This particular
    module assumes that the transformations map a complex distribution to the standard Normal
    distribution. Its :meth:`forward` method then yields the log-probability in the target
    distribution.
    """

    def __init__(self, transforms):
        """
        Initializes a new normalizing flow applying the given transformations.

        Parameters
        ----------
        transforms: list of torch.nn.Module
            Transformations whose :code:`forward` method yields the transformed value and the log-
            determinant of the applied transformation. All transformations must have the same
            dimension.
        """
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, z):
        """
        Computes the log-probabilities for the given samples after applying this flow's
        transformations.

        Parameters
        ----------
        z: torch.Tensor [N, D]
            The input value (batch size N, dimensionality D).

        Returns
        -------
        torch.Tensor [N]
            The log-probabilities for all given samples.
        """
        batch_size = z.size(0)
        dim = z.size(1)
        device = z.device

        log_det_sum = torch.zeros(batch_size, device=device)
        for transform in self.transforms:
            z, log_det = transform(z)
            log_det_sum += log_det

        log_prob_normal = -0.5 * (dim * math.log(2 * math.pi) + (z * z).sum(-1))
        log_prob = log_prob_normal + log_det_sum

        return log_prob
