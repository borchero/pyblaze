import torch.nn as nn

class LinearResidual(nn.Module):
    """
    Residual module that models a two-layer MLP with nonlinearity and adds the input to the output:

        .. math::

            f(x) = x + W_2 \\sigma(W_1 x + b_1) + b_2

    Usually, another nonlineary is applied to the output.
    """

    def __init__(self, dim, hidden_dim, activation=nn.ReLU(), bias=True):
        """
        Initializes a new residual module.

        Parameters
        ----------
        dim: int
            The dimension of the input. Equals the dimension of the output.
        hidden_dim: int
            The hidden dimension (i.e. the output dimension of :math:`W_1`).
        activation: torch.nn.Module, default: torch.nn.ReLU()
            An activation function to use (i.e. :math:`\\sigma` in the formula above).
        bias: bool, default: True
            Whether to add biases to the linear layers (i.e. :math:`b_{12}` in the formula above).
        """
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.activation = activation
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):
        """
        Computes the output of the residual module.

        Parameters
        ----------
        x: torch.Tensor [N, D]
            The input (batch size N, dimensionality D).

        Returns
        -------
        torch.Tensor [N, D]
            The processed output.
        """
        z = self.activation(self.w1(x))
        return x + self.w2(z)
