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

    def __init__(self, *dims, activation=nn.LeakyReLU(), seed=0, permute=False):
        """
        Initializes a new MADE model as a sequence of masked linear layers.

        Parameters
        ----------
        dims: varargs of int
            Dimensions of input (first), output (last) and hidden layers. At least one hidden layer
            must be defined, i.e. at least 3 dimensions must be given. The output dimension must be
            equal to the input dimension or a multiple of it.
        activation: torch.nn.Module, default: torch.nn.LeakyReLU()
            An activation function to be used after linear layers (except for the output layer).
            This module is shared for all hidden layers.
        seed: int, default: None
            A seed to use for initializing the random number generator for constructing random
            masks for the hidden layers. If set to `None`, deterministic initialization is used.
        permute: bool, default: False
            Whether to arbitrarily permute the input (permutation is applied deterministically).
        """
        super().__init__()

        if len(dims) < 3:
            raise ValueError("MADE model must have at least one hidden layer")
        if dims[-1] % dims[0] != 0:
            raise ValueError("Output dimension must be multiple of the input dimension")

        self.dims = dims

        if seed is None:
            m_layers = _generate_sequential(dims)
        else:
            generator = torch.Generator().manual_seed(seed)
            m_layers = _generate_random_numbers(dims, generator, permute)

        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
            if i > 0:
                layers.append(activation)

            hidden = i < len(dims) - 2
            mask = _generate_mask(m_layers[i], m_layers[i+1], hidden=hidden)
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

    degrees = [torch.arange(in_dim)]
    for dim in dims[1:]:
        degrees += [torch.arange(dim) % (in_dim - 1)]
    degrees += [torch.arange(in_dim) % in_dim - 1]

    return degrees


def _generate_random_numbers(dims, generator, permute):
    in_dim = dims[0]

    samples = []
    # Avoid unconnected units by sampling at least the minimum number of connected neurons in the
    # previous layer
    min_val = 0

    # We assign values between 0 and D-2 such that we can simply arange/permute the indices for the
    # input layer
    for i, dim in enumerate(dims[:-1]):
        if i == 0:
            m_vals = torch.randperm(dim, generator=generator) if permute else torch.arange(dim)
        else:
            m_vals = torch.randint(min_val, in_dim-1, size=(dim,), generator=generator)
            min_val = m_vals.min().item()
        samples.append(m_vals)

    if dims[-1] > dims[0]:
        samples.append(samples[0].repeat(dims[-1] // dims[0]))
    else:
        samples.append(samples[0])

    return samples


def _generate_mask(m_prev, m_next, hidden=True):
    if hidden:
        mask = m_next[None, :] >= m_prev[:, None]
    else:  # for output layer
        mask = m_next[None, :] > m_prev[:, None]
    return mask.float().t()
