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
        nll = -X.log_prob_standard_normal(z) - log_det
        if self.reduction == 'mean':
            return nll.mean()
        if self.reduction == 'sum':
            return nll.sum()
        return nll


class TransformedGmmLoss(nn.Module):
    """
    This loss returns the negative log-likelihood (NLL) of some data that has been transformed via
    invertible transformations. The NLL is computed via the negative sum of the log-determinant of
    the transformations and the log-probability of observing the output under a GMM with predefined
    means and unit variances. The simple alternative to this loss is the
    :class:`TransformedNormalLoss`.
    """

    def __init__(self, means, trainable=False, reduction='mean'):
        """
        Initializes a new GMM loss.

        Parameters
        ----------
        means: torch.Tensor [N, D]
            The means of the GMM. For random initialization of the means, consider using
            :meth:`pyblaze.nn.functional.random_gmm`.
        trainable: bool, default: False
            Whether the means are trainable.
        reduction: str, default: 'mean'
            The kind of reduction to perform. Must be one of ['mean', 'sum', 'none'].
        """
        super().__init__()
        if trainable:
            self.means = nn.Parameter(means)
        else:
            self.register_buffer('means', means)
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
        nll = -X.log_prob_standard_gmm(z, self.means) - log_det
        if self.reduction == 'mean':
            return nll.mean()
        if self.reduction == 'sum':
            return nll.sum()
        return nll
