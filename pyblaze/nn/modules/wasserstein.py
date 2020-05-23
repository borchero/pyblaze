import torch.nn as nn

class WassersteinLossGenerator(nn.Module):
    """
    Computes the loss of the generator in the Wasserstein GAN setting.
    """

    def forward(self, out):
        """
        Computes the loss for the generator given the outputs of the critic.

        Parameters
        ----------
        out: torch.Tensor [N]
            The output values of the critic (batch size N).

        Returns
        -------
        torch.Tensor [1]
            The loss incurred for the generator.
        """
        return -out.mean()


class WassersteinLossCritic(nn.Module):
    """
    Computes the loss of the critic in the Wasserstein GAN setting. This loss optionally includes
    a gradient penalty that should be used if no other regularization methods (weight clipping,
    spectral normalization, ...) are used.
    """

    def __init__(self, gradient_penalty=None):
        """
        Initializes a new Wasserstein loss for a critic.

        Parameters
        ----------
        gradient_penalty: nn.Module, default: False
            A gradient penalty object that accepts fake and real inputs to the critic and computes
            the gradient penalty for it.
        """
        super().__init__()
        self.gradient_penalty = gradient_penalty

    def forward(self, out_fake, out_real, *inputs):
        """
        Computes the loss for the critic given the outputs of itself and potentially a tuple of
        inputs.

        Parameters
        ----------
        out_fake: torch.Tensor [N]
            The critic's output for the fake inputs (batch size N).
        out_real: torch.Tensor [N]
            The critic's output for the real inputs.
        inputs: tuple of (torch.Tensor [N, ...], torch.Tensor [N, ...])
            A tuple of `(in_fake, in_real)` that must be given if a gradient penalty is used.

        Returns
        -------
        torch.Tensor [1]
            The loss incurred for the critic.
        torch.Tensor [1]
            The estimated Earth mover's (Wasserstein-1) distance (equal to the detached negative
            loss if there is no gradient penalty).
        """
        loss = out_fake.mean() - out_real.mean()
        wass_dist = -loss.detach()

        penalty = 0
        if self.gradient_penalty is not None:
            penalty = self.gradient_penalty(*inputs)

        return loss + penalty, wass_dist
