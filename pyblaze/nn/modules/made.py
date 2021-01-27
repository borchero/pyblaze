import torch
import torch.nn as nn
import torch.nn.functional as F

class MADE(nn.Module):
    """
    Masked autoencoder for distribution estimation (MADE) as introduced in
    `MADE: Masked Autoencoder for Distribution Estimation <https://arxiv.org/abs/1502.03509>`_
    (Germain et al., 2015). In consists of a series of masked linear layers and a given
    non-linearity between them.
    """

    def __init__(self, *dims, activation=nn.LeakyReLU()):
        """
        Initializes a new MADE model as a sequence of masked linear layers.

        Parameters
        ----------
        dims: varargs of int
            Dimensions of input (first), output (last) and hidden layers. At least one hidden layer
            must be defined, i.e. at least 3 dimensions must be given. The output dimension must be
            equal to the input dimension or a multiple of it. Hidden dimensions should be a
            multiple of the input dimension unless a seed for random initialization is given.
        activation: torch.nn.Module, default: torch.nn.LeakyReLU()
            An activation function to be used after linear layers (except for the output layer).
            This module is shared for all hidden layers.
        """
        super().__init__()

        if len(dims) < 3:
            raise ValueError("MADE model must have at least one hidden layer")
        if dims[-1] % dims[0] != 0:
            raise ValueError("Output dimension must be multiple of the input dimension")

        self.dims = dims
        m_layers = _generate_sequential(dims)

        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
            if i > 0:
                layers.append(activation)

            mask = (m_layers[i+1].unsqueeze(-1) >= m_layers[i].unsqueeze(0)).float()
            layers.append(_MaskedLinear(in_dim, out_dim, mask=mask))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Computes the outputs of the MADE model.

        Parameters
        ----------
        x: torch.Tensor [..., D]
            The input (input dimension D).

        Returns
        -------
        torch.Tensor [..., E]
            The output (output dimension E).
        """
        return self.mlp(x)


class _MaskedLinear(nn.Linear):

    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', mask)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)

    def __repr__(self):
        return f'MaskedLinear(in_features={self.in_features}, ' + \
            f'out_features={self.out_features}, bias={self.bias is not None})'


def _generate_sequential(dims):
    in_dim = dims[0]
    out_dim = dims[-1]

    degrees = [torch.arange(in_dim) + 1]
    for dim in dims[1:-1]:
        degrees += [torch.arange(dim) % (in_dim - 1) + 1]
    degrees += [torch.arange(out_dim) % in_dim]

    return degrees
