import torch.nn as nn

class View(nn.Module):
    """
    Utility module that views the input as a new dimension. This module is usually used when
    making use of :code:`torch.nn.Sequential` and requiring reshaping a linear output layer to a 2D
    input or the like.
    """

    def __init__(self, *dim):
        """
        Initializes a new view module.

        Parameters
        ----------
        dim: varargs of int
            The new dimension. May contain no more than one -1.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Views the input as this module's view dimension.

        Parameters
        ----------
        x: torch.Tensor
            The tensor to view differently.

        Returns
        -------
        torch.Tensor
            The input tensor with a new view on it.
        """
        return x.view(*self.dim)
