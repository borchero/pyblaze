import torch.nn as nn
import pyblaze.nn.functional as X

class TransformedNormalLoss(nn.Module):
    """
    This loss returns the negative log-likelihood (NLL) of some data that has been transformed via
    invertible transformations. The NLL is computed via the negative sum of the log-determinant of
    the transformations and the log-probability of observing the output under a standard Normal
    distribution. This loss is typically used to fit a normalizing flow.
    """

    def __init__(self, reduction='mean'):
        """
        Initializes a new NLL loss.

        Parameters
        ----------
        reduction: str, default: 'mean'
            The kind of reduction to perform. Must be one of ['mean', 'sum', 'none'].
        """
        super().__init__()

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"Invalid reduction {reduction}")

        self.reduction = reduction

    def forward(self, z, log_det):
        """
        Computes the NLL for the given transformed values.

        Parameters
        ----------
        z: torch.Tensor [N, D]
            The output values of the transformations (batch size N, dimensionality D).
        log_det: torch.Tensor [N]
            The log-determinants of the transformations for all values.

        Returns
        -------
        torch.Tensor [1]
            The mean NLL for all given values.
        """
        result = -X.log_prob_standard_normal(z) - log_det
        if self.reduction == 'mean':
            return result.mean()
        if self.reduction == 'sum':
            return result.sum()
        return result
