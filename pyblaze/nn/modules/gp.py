import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class GradientPenalty(nn.Module):
    """
    Implementation of the gradient penalty as presented in "Improved Training of Wasserstein GANs"
    (Gulrajani et al., 2017). It ensures that the norm of the critic's gradient is close to 1,
    ensuring Lipschitz continuity.

    Optionally, the gradient penalty can be replaced by a Lipschitz penalty which does not penalize
    gradients smaller than one. It is taken from "On the Regularization of Wasserstein GANs"
    (Petzka et al., 2018).
    """

    def __init__(self, module, coefficient=10, lipschitz=False):
        """
        Initializes a new gradient penalty for the given module.

        Parameters
        ----------
        module: torch.nn.Module
            The module whose gradient norm should be penalized.
        coefficient: float, default: 10
            The coefficient for the gradient penalty. The default value is taken from the original
            WGAN-GP paper.
        lipschitz: boolean, default: False
            Whether to use Lipschitz penalty instead of simple gradient penalty (not penalizing
            gradient norms smaller than 1).
        """
        super().__init__()

        self.module = module
        self.coefficient = coefficient
        self.lipschitz = lipschitz

    def forward(self, fake, real):
        """
        Computes the loss incurred on the penalized module based on a batch of fake and real
        instances.

        Parameters
        ----------
        fake: torch.Tensor [N, ...]
            The fake instances (batch size N).
        real: torch.Tensor [N, ...]
            The real instances.

        Returns
        -------
        torch.Tensor [1]
            The gradient penalty times the penalty coefficient.
        """
        interpolation, out = self.interpolate(fake, real)

        grad_out = torch.ones_like(out).requires_grad_(False)
        gradients = autograd.grad(
            out, interpolation, grad_outputs=grad_out, create_graph=True, retain_graph=True
        )[0]
        gradients = gradients.contiguous().view(gradients.size(0), -1)

        target = gradients.norm(2, dim=1) - 1
        if self.lipschitz:
            target = F.relu(target)
        return self.coefficient * (target ** 2).mean()

    def interpolate(self, fake, real):
        """
        Interpolates the given fake and real instances with an arbitrary alpha value weighing each
        batch sample. By default, it assumes that fake and real instances can be interpolated over
        the first dimension. This method may be overridden by subclasses for more complicated
        models.

        Parameters
        ----------
        fake: torch.Tensor [N, ...]
            The fake instances passed to the module (batch size N).
        real: torch.Tensor [N, ...]
            The real instances passed to the module.

        Returns
        -------
        torch.Tensor [N, ...]
            The interpolation which (which must have `requires_grad` set to `True`).
        torch.Tensor [N]
            The module's output for the interpolated fake and real instances.
        """
        batch_size = fake.size(0)
        dim = fake.dim()

        alpha = fake.new(batch_size, *([1] * (dim - 1))).uniform_(0.0, 1.0)
        interpolation = alpha * fake + (1 - alpha) * real
        interpolation.requires_grad_()

        return interpolation, self.module(interpolation)
