import torch
import torch.nn as nn

class NormalizingFlow(nn.Module):
    """
    In general, a normalizing flow is a module to transform an initial density into another one
    (usually a more complex one) via a sequence of invertible transformations.
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

    def forward(self, z, condition=None):
        """
        Computes the outputs and log-detemrinants for the given samples after applying this flow's
        transformations.

        Parameters
        ----------
        z: torch.Tensor [N, D]
            The input value (batch size N, dimensionality D).
        condition: torch.Tensor [N, C]
            An additional condition vector on which the transforms are conditioned. Causes failure
            if any of the underlying transforms does not support conditioning.

        Returns
        -------
        torch.Tensor [N, D]
            The transformed values.
        torch.Tensor [N]
            The log-determinants of the transformation for all values.
        """
        batch_size = z.size(0)
        device = z.device
        kwargs = {'condition': condition} if condition is not None else {}

        log_det_sum = torch.zeros(batch_size, device=device)
        for transform in self.transforms:
            z, log_det = transform(z, **kwargs)
            log_det_sum += log_det

        return z, log_det_sum
