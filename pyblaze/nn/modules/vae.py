import torch.nn as nn

class VAELoss(nn.Module):
    """
    Loss for the reconstruction error of a variational autoencoder when the encoder parametrizes a
    Gaussian distribution. Taken from "Auto-Encoding Variational Bayes" (Kingma and Welling, 2014).
    """

    def __init__(self, loss, kld_coefficient=1):
        """
        Initializes a new loss for a variational autoencoder.

        Parameters
        ----------
        loss: torch.nn.Module
            The loss to incur for the decoder's output given `(x_pred, x_true)`. This might e.g. be
            a BCE loss.
        kld_coefficient: float, default: 1
            The multiplier for the Kullback-Leibler divergence that enforces that the encoder
            outputs values from a Gaussian distribution.
        """
        super().__init__()

        self.loss = loss
        self.kld_coef = kld_coefficient

    def forward(self, x_pred, x_true, mu, logvar):
        """
        Computes the loss of the decoder's output.

        Parameters
        ----------
        x_pred: torch.Tensor [N, ...]
            The outputs of the decoder (batch size N).
        x_true: torch.Tensor [N, ...]
            The target outputs of the decoder.
        mu: torch.Tensor [N, D]
            The output for the means from the encoder (dimensionality D).
        logvar: torch.Tensor [N, D]
            The output for the log-values of the diagonal entries of the covariance matrix.

        Returns
        -------
        torch.Tensor [1]
            The loss incurred computed as the actual loss plus a weighted KL-divergence.
        """
        loss = self.loss(x_pred, x_true)
        kld = -0.5 * (1 + logvar - mu * mu - logvar.exp()).mean()
        return loss + self.kld_coef * kld
