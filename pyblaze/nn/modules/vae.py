import torch.nn as nn

class VAELoss(nn.Module):
    """
    Loss for the reconstruction error of a variational autoencoder when the encoder parametrizes a
    Gaussian distribution. Taken from "Auto-Encoding Variational Bayes" (Kingma and Welling, 2014).
    """

    def __init__(self, loss):
        """
        Initializes a new loss for a variational autoencoder.

        Parameters
        ----------
        loss: torch.nn.Module
            The loss to incur for the decoder's output given `(x_pred, x_true)`. This might e.g. be
            a BCE loss. **The reduction must be 'none'.**
        """
        super().__init__()
        self.loss = loss

    def forward(self, x_pred, mu, logvar, x_true):
        """
        Computes the loss of the decoder's output.

        Parameters
        ----------
        x_pred: torch.Tensor [N, ...]
            The outputs of the decoder (batch size N).
        mu: torch.Tensor [N, D]
            The output for the means from the encoder (dimensionality D).
        logvar: torch.Tensor [N, D]
            The output for the log-values of the diagonal entries of the covariance matrix.
        x_true: torch.Tensor [N, ...]
            The target outputs for the decoder.

        Returns
        -------
        torch.Tensor [1]
            The loss incurred computed as the actual loss plus a weighted KL-divergence.
        """
        dims = range(1, x_pred.dim())  # we want to sum over all dimensions but the batch dimension
        loss = self.loss(x_pred, x_true).sum(tuple(dims))
        kld = -0.5 * (1 + logvar - mu * mu - logvar.exp()).sum(-1)
        return (loss + kld).mean()
