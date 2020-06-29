import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

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
    following function for :math:`\mathbf{z} \in \mathbb{R}^D`:

    .. math::

        f_{\mathbf{a}, \mathbf{b}}(\mathbf{z}) = \mathbf{a} \odot \mathbf{z} + \mathbf{b}


    with :math:`\mathbf{a} \in \mathbb{R}^D_+` and :math:`\mathbf{b} \in \mathbb{R}^D`.

    The log-determinant of its Jacobian is given as follows:

    .. math::

        \sum_{k=1}^D{\log{a_k}}


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

        log_det = self.log_alpha.sum()  # [1]
        log_det = log_det.expand(batch_size)  # [N]

        return y, log_det

################################################################################
### PLANAR TRANSFORM
################################################################################

class PlanarTransform(_Transform):
    r"""
    A planar transformation may be used to split the input along a hyperplane. It was introduced in
    "Variational Inference with Normalizing Flows" (Rezende and Mohamed, 2015). It computes the
    following function for :math:`\mathbf{z} \in \mathbb{R}^D` (although the planar transform was
    introduced for an arbitrary activation function :math:`\sigma`, this transform restricts the
    usage to :math:`tanh`):

    .. math::

        f_{\mathbf{u}, \mathbf{w}, b}(\mathbf{z}) =
            \mathbf{z} + \mathbf{u} \tanh(\mathbf{w}^T \mathbf{z} + b)


    with :math:`\mathbf{u}, \mathbf{w} \in \mathbb{R}^D` and :math:`b \in \mathbb{R}`.

    The log-determinant of its Jacobian is given as follows:

    .. math::

        \log\left| 1 + \mathbf{u}^T ((1 - \tanh^2(\mathbf{w}^T \mathbf{z} + b))\mathbf{w}) \right|


    This transform is invertible for its outputs.
    """

    def __init__(self, dim):
        r"""
        Initializes a new planar transformation.

        Parameters
        ----------
        dim: int
            The dimension of the inputs to the function.
        """
        super().__init__(dim)

        self.u = nn.Parameter(torch.empty(dim))
        self.w = nn.Parameter(torch.empty(dim))
        self.bias = nn.Parameter(torch.empty(1))
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
        dot = self.u @ self.w  # [1]
        m = F.softplus(dot) - 1  # [1]
        w_prime = self.w / (self.w ** 2).sum()  # [D]
        u_prime = self.u + (m - dot) * w_prime  # [D]

        sigma = (z @ self.w + self.bias).tanh()  # [N]
        y = z + sigma.ger(u_prime)  # [N, D]

        sigma_d = 1 - sigma ** 2  # [N]
        phi = sigma_d.ger(self.w)  # [N, D]
        det = (1 + phi @ u_prime).abs()  # [N]
        log_det = det.log()  # [N]

        return y, log_det

################################################################################
### RADIAL TRANSFORM
################################################################################

class RadialTransform(_Transform):
    r"""
    A radial transformation may be used to apply radial contractions and expansions around a
    reference point. It was introduced in "Variational Inference with Normalizing Flows" (Rezende
    and Mohamed, 2015). It computes the following function for :math:`\mathbf{z} \in \mathbb{R}^D`:

    .. math::

        f_{\mathbf{z}_0, \alpha, \beta}(\mathbf{z}) =
            \mathbf{z} + \beta h(\alpha, r) (\mathbf{z} - \mathbf{z}_0)


    with :math:`\mathbf{z}_0 \in \mathbb{R}^D`, :math:`\alpha \in \mathbb{R}^+`,
    :math:`\beta \in \mathbb{R}`, :math:`\mathbf{r} = ||\mathbf{z} - \mathbf{z}_0||_2` and
    :math:`h(\alpha, r) = (\alpha + r)^{-1}`.

    The log-determinant of its Jacobian is given as follows:

    .. math::

        (D - 1) \log\left(1 + \beta h(\alpha, r)\right) +
            \log\left(1 + \beta h(\alpha, r) - \beta h^2(\alpha, r) r \right)


    This transform is invertible for its outputs, however, there does not exist a closed-form
    solution for computing the inverse in general.
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
        self.alpha_prime = nn.Parameter(torch.empty(1))
        self.beta_prime = nn.Parameter(torch.empty(1))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets this module's parameters. All parameters are sampled from a standard Normal
        distribution.
        """
        std = 1 / math.sqrt(self.reference.size(0))
        init.uniform_(self.reference, -std, std)
        init.uniform_(self.alpha_prime, -std, std)
        init.uniform_(self.beta_prime, -std, std)

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
        alpha = F.softplus(self.alpha_prime)  # [1]
        beta = -alpha + F.softplus(self.beta_prime)  # [1]

        diff = z - self.reference  # [N, D]
        r = diff.norm(dim=-1, keepdim=True)  # [N, 1]
        h = (alpha + r).reciprocal()  # [N]
        beta_h = beta * h  # [N]
        y = z + beta_h * diff  # [N, D]

        h_d = -(h ** 2)  # [N]
        log_det_lhs = (self.dim - 1) * beta_h.log1p()  # [N]
        log_det_rhs = (beta_h + beta * h_d * r).log1p()  # [N, 1]
        log_det = (log_det_lhs + log_det_rhs).view(-1)  # [N]

        return y, log_det

################################################################################
### AFFINE COUPLING
################################################################################

class AffineCouplingTransform1d(_Transform):
    r"""
    An affine coupling transforms the input by splitting it into two parts and transforming the
    second part by an arbitrary function depending on the first part. It was introduced in
    "Density Estimation Using Real NVP" (Dinh et. al, 2017). It computes the following function for
    :math:`\mathbf{z} \in \mathbb{R}^D` and a dimension :math:`d < D`:

    .. math::

        f_{\mathbf{\omega}_s, \mathbf{\omega}_m}(\mathbf{z}) =
            [\mathbf{z}_{1:d}, \mathbf{z}_{d+1:D} \odot
            \exp(g_{\mathbf{\omega}_s}(\mathbf{z}_{1:d})) +
            h_{\mathbf{\omega}_m}(\mathbf{z}_{1:d})]^T


    with :math:`g, h: \mathbb{R}^d \rightarrow \mathbb{R}^{D-d}` being arbitrary parametrized
    functions (e.g. neural networks) computing the log-scale and the translation, respectively.

    The log-determinant of its Jacobian is given as follows:

    .. math::

        \sum_{k=1}^{D-d}{g_{\mathbf{\omega}_s}(\mathbf{z}_{1:d})}


    Additionally, this transform can be easily conditioned on another input variable
    :math:`\mathbf{x}` by conditioning the functions :math:`g, h` on it.
    This transform is invertible and the inverse computation will be added in the future.

    Note
    ----
    As only part of the input is transformed, consider using this class with the :code:`reverse`
    flag set alternately.
    """

    def __init__(self, dim, fixed_dim, net, reverse=False):
        """
        Initializes a new affine coupling transformation.

        Parameters
        ----------
        dim: int
            The dimensionality of the input.
        fixed_dim: int
            The dimensionality of the input space that is not transformed. Must be smaller than the
            dimension.
        net: torch.nn.Module [N, F] -> ([N, A], [N, A])
            An arbitrary neural network taking as input the fixed part of the input and outputting
            a mean and a log scale used for scaling and translating the affine part of the input,
            respectively. In case this affine coupling is used with conditioning, the net's input
            dimension should be modified accordingly (batch size N, fixed dimension F, affine
            dimension A).
        reverse: bool, default: False
            Whether to keep the second part fixed instead of the first one.
        """
        super().__init__(dim)

        if fixed_dim >= dim:
            raise ValueError("fixed_dim must be smaller than dim")

        self.fixed_dim = fixed_dim
        self.reverse = reverse

        self.net = net

    def forward(self, z, condition=None):
        """
        Transforms the given input.

        Parameters
        ----------
        z: torch.Tensor [N, D]
            The given input (batch size N, dimensionality D).
        condition: torch.Tensor [N, C]
            An optional tensor on which this layer's net is conditioned. This value will be
            concatenated with the part of :code:`z` that is passed to this layer's net.

        Returns
        -------
        torch.Tensor [N, D]
            The transformed input.
        torch.Tensor [N]
            The log-determinants of the Jacobian evaluated at z.
        """
        if self.reverse:
            z2, z1 = z.split(self.fixed_dim, dim=-1)
        else:
            z1, z2 = z.split(self.fixed_dim, dim=-1)

        if condition is None:
            x = z1
        else:
            x = torch.cat([z1, condition], dim=1)

        mean, logscale = self.net(x)
        logscale = logscale.tanh()
        transformed = z2 * logscale.exp() + mean

        if self.reverse:
            y = torch.cat([transformed, z1], dim=-1)
        else:
            y = torch.cat([z1, transformed], dim=-1)

        log_det = logscale.sum(-1)

        return y, log_det
