import torch
import torch.nn as nn
import torch.nn.init as init
import torch.autograd as autograd

# pylint: disable=abstract-method
class _Transform(nn.Module):
    """
    A base class for all transforms.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.dim})'

################################################################################
### AFFINE TRANSFORM
################################################################################

class AffineTransform(_Transform):
    r"""
    An affine transformation may be used to transform an input variable linearly. It computes the
    following function for :math:`\bm{z} \in \mathbb{R}^D`:

        :math:`f_{\bm{a}, \bm{b}}(\bm{z}) = \bm{a} \odot \bm{z} + \bm{b}`

    with :math:`\bm{a} \in \mathbb{R}^D_+` and :math:`\bm{b} \in \mathbb{R}^D`.

    The log-determinant of its Jacobian is given as follows:

        :math:`\sum_{k=1}^D{\log{a_k}}`

    Although this transformation is theoretically invertible, the inverse function is not
    implemented at the moment.
    """

    def __init__(self, dim):
        """
        Initializes a new affine transformation.

        Parameters
        ----------
        dim: int
            The dimension of the inputs to the function.
        """
        super().__init__(dim)

        self.log_alpha = nn.Parameter(torch.empty(dim))
        self.beta = nn.Parameter(torch.empty(dim))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets this module's parameters. All parameters are sampled uniformly from [0, 1].
        """
        init.uniform_(self.log_alpha)
        init.uniform_(self.beta)

    def forward(self, z):
        """
        Transforms the given input.

        Parameters
        ----------
        z: torch.Tensor [N, D]
            The given input (batch size N, transform dimensionality D).

        Returns
        -------
        torch.Tensor [N, D]
            The transformed input.
        torch.Tensor [N]
            The log-determinants of the Jacobian evaluated at the input.
        """
        batch_size = z.size(0)

        y = self.log_alpha.exp() * z + self.beta  # [N, D]

        log_det = self.alpha.sum()  # [1]
        log_det = log_det.expand(batch_size)  # [N]

        return y, log_det

################################################################################
### PLANAR TRANSFORM
################################################################################

class PlanarTransform(_Transform):
    r"""
    A planar transformation may be used to split the input along a hyperplane. It was introduced in
    "Variational Inference with Normalizing Flows" (Rezende and Mohamed, 2015). It computes the
    following function for :math:`\bm{z} \in \mathhbb{R}^D` where :math:`\\sigma` is an arbitrary
    activation function:

        :math:`f_{\bm{u}, \bm{w}, b}(\bm{z}) = \bm{z} + \bm{u} \sigma(\bm{w}^T \bm{z} + b)`

    with :math:`\bm{u}, \bm{w} \in \mathhbb{R}^D` and :math:`b \in \mathbb{R}`.

    The log-determinant of its Jacobian is given as follows:

        :math:`\log\left| 1 + \bm{u}^T (\sigma'(\bm{w}^T \bm{z} + b) \bm{w}) \right|`

    This transform is implemented without ensuring invertibility.
    """

    def __init__(self, dim, activation=nn.Tanh()):
        r"""
        Initializes a new planar transformation.

        Parameters
        ----------
        dim: int
            The dimension of the inputs to the function.
        activation: torch.nn.Module, default: torch.nn.Tanh()
            The activation function to use. By default, :math:`\tanh` is used.
        """
        super().__init__(dim)

        self.u = nn.Parameter(torch.empty(dim))
        self.w = nn.Parameter(torch.empty(dim))
        self.bias = nn.Parameter(torch.empty(1))
        self.activation = activation

        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets this module's parameters. All parameters are sampled uniformly from [0, 1].
        """
        init.uniform_(self.u)
        init.uniform_(self.w)
        init.uniform_(self.bias)

    def forward(self, z):
        """
        Transforms the given input.

        Parameters
        ----------
        z: torch.Tensor [N, D]
            The given input (batch size N, transform dimensionality D).

        Returns
        -------
        torch.Tensor [N, D]
            The transformed input.
        torch.Tensor [N]
            The log-determinants of the Jacobian evaluated at z.
        """
        inner = z @ self.w + self.bias  # [N]
        sigma = self.activation(inner)  # [N]
        y = z + sigma.ger(self.u)  # [N, D]

        sigma_d = autograd.grad(sigma.sum(), inner, create_graph=True)  # [N]
        phi = sigma_d.ger(self.w)  # [N, D]
        det = (1 + phi @ self.u).abs()  # [N]
        log_det = det.log()  # [N]

        return y, log_det

################################################################################
### RADIAL TRANSFORM
################################################################################

class RadialTransform(_Transform):
    r"""
    A radial transformation may be used to apply radial contractions and expansions around a
    reference point. It was introduced in "Variational Inference with Normalizing Flows" (Rezende
    and Mohamed, 2015). It computes the following function for :math:`\bm{z} \in \mathhbb{R}^D`:

        :math:`f_{\bm{z}_0, \alpha, \beta}(\bm{z}) = \bm{z} + \beta h(\alpha, r)(\bm{z} - \bm{z}_0)`

    with :math:`\bm{z}_0 \in \mathbb{R}^D`, :math:`\alpha \in \mathbb{R}^+`,
    :math:`\beta \in \mathbb{R}`, :math:`\bm{r} = ||\bm{z} - \bm{z}_0||_2` and
    :math:`h(\alpha, r) = (\alpha + r)^{-1}`.

    The log-determinant of its Jacobian is given as follows:

        :math:`\log\left| 1 + \bm{u}^T (\sigma'(\bm{w}^T \bm{z} + b) \bm{w}) \right|`

    This transform is implemented without ensuring invertibility.
    """

    def __init__(self, dim):
        r"""
        Initializes a new planar transformation.

        Parameters
        ----------
        dim: int
            The dimension of the inputs to the function.
        activation: torch.nn.Module, default: torch.nn.Tanh()
            The activation function to use. By default, :math:`\tanh` is used.
        """
        super().__init__(dim)

        self.reference = nn.Parameter(torch.empty(dim))
        self.log_alpha = nn.Parameter(torch.empty(1))
        self.beta = nn.Parameter(torch.empty(1))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets this module's parameters. All parameters are sampled from a standard Normal
        distribution.
        """
        init.normal_(self.reference)
        init.normal_(self.log_alpha)
        init.normal_(self.beta)

    def forward(self, z):
        """
        Transforms the given input.

        Parameters
        ----------
        z: torch.Tensor [N, D]
            The given input (batch size N, transform dimensionality D).

        Returns
        -------
        torch.Tensor [N, D]
            The transformed input.
        torch.Tensor [N]
            The log-determinants of the Jacobian evaluated at z.
        """
        alpha = self.log_alpha.exp()  # [1]

        diff = z - self.reference  # [N, D]
        r = diff.norm(dim=-1, keepdim=True)  # [N, 1]
        h = (alpha + r).reciprocal()  # [N]
        beta_h = self.beta * h  # [N]
        y = z + beta_h * diff  # [N, D]

        h_d = -(h ** 2)  # [N]
        log_det_lhs = (self.dim - 1) * beta_h.log1p()  # [N]
        log_det_rhs = (beta_h + self.beta * h_d * r).log1p()  # [N, 1]
        log_det = (log_det_lhs + log_det_rhs).view(-1)  # [N]

        return y, log_det
