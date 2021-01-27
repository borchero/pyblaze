import math
import torch
import torch.nn as nn
import torch.nn.init as nninit
import torch.nn.functional as F
from .made import MADE

# pylint: disable=abstract-method
class _Transform(nn.Module):
    """
    A base class for all transforms.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __repr__(self):
        if self.dim is None:
            return f'{self.__class__.__name__}()'
        return f'{self.__class__.__name__}(dim={self.dim})'

#--------------------------------------------------------------------------------------------------

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
        nninit.uniform_(self.log_alpha)
        nninit.uniform_(self.beta)

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

#--------------------------------------------------------------------------------------------------

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
        std = 1 / math.sqrt(self.u.size(0))
        nninit.uniform_(self.u, -std, std)
        nninit.uniform_(self.w, -std, std)
        nninit.uniform_(self.bias, -std, std)

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

#--------------------------------------------------------------------------------------------------

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
        nninit.uniform_(self.reference, -std, std)
        nninit.uniform_(self.alpha_prime, -std, std)
        nninit.uniform_(self.beta_prime, -std, std)

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

#--------------------------------------------------------------------------------------------------

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

    def __init__(self, dim, fixed_dim, net, constrain_scale=False):
        """
        Initializes a new affine coupling transformation.

        Parameters
        ----------
        dim: int
            The dimensionality of the input.
        fixed_dim: int
            The dimensionality of the input space that is not transformed. Must be smaller than the
            dimension.
        net: torch.nn.Module [N, F] -> [N, F*2]
            An arbitrary neural network taking as input the fixed part of the input and outputting
            a mean and a log scale used for scaling and translating the affine part of the input,
            respectively, as a single tensor which will be split. In case this affine coupling is
            used with conditioning, the net's input dimension should be modified accordingly (batch
            size N, fixed dimension F).
        constrain_scale: bool, default: False
            Whether to constrain the scale parameter that the output is multiplied by. This should
            be set for deep normalizing flows where no batch normalization is used.
        """
        super().__init__(dim)

        if fixed_dim >= dim:
            raise ValueError("fixed_dim must be smaller than dim")

        self.fixed_dim = fixed_dim
        self.constrain_scale = constrain_scale
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
            concatenated with the part of :code:`z` that is passed to this layer's net (condition
            dimension C).

        Returns
        -------
        torch.Tensor [N, D]
            The transformed input.
        torch.Tensor [N]
            The log-determinants of the Jacobian evaluated at z.
        """
        z1, z2 = z.split(self.fixed_dim, dim=-1)

        if condition is None:
            x = z1
        else:
            x = torch.cat([z1, condition], dim=1)

        mean, logscale = self.net(x).chunk(2, dim=1)
        if self.constrain_scale:
            logscale = logscale.tanh()
        transformed = z2 * logscale.exp() + mean

        y = torch.cat([z1, transformed], dim=-1)
        log_det = logscale.sum(-1)
        return y, log_det

#--------------------------------------------------------------------------------------------------

class MaskedAutoregressiveTransform1d(_Transform):
    r"""
    1-dimensional Masked Autogressive Transform as introduced in
    `Masked Autoregressive Flow for Density Estimation <https://arxiv.org/abs/1705.07057>`_
    (Papamakarios et al., 2018).
    """

    def __init__(self, dim, *hidden_dims, activation=nn.LeakyReLU(), constrain_scale=False):
        """
        Initializes a new MAF transform that is backed by a :class:`pyblaze.nn.MADE` model.

        Parameters
        ----------
        dim: int
            The dimension of the inputs.
        hidden_dims: varargs of int
            The hidden dimensions of the MADE model.
        activation: torch.nn.Module, default: torch.nn.LeakyReLU()
            The activation function to use in the MADE model.
        constrain_scale: bool, default: False
            Whether to constrain the scale parameter that the output is multiplied by. This should
            be set for deep normalizing flows where no batch normalization is used.
        """
        super().__init__(dim)

        self.constrain_scale = constrain_scale
        self.net = MADE(dim, *hidden_dims, dim * 2, activation=activation)

    def forward(self, x):
        """
        Transforms the given input.

        Parameters
        ----------
        z: torch.Tensor [N, D]
            The given input (batch size N, dimensionality D).

        Returns
        -------
        torch.Tensor [N, D]
            The transformed input.
        torch.Tensor [N]
            The log-determinants of the Jacobian evaluated at z.
        """
        mean, logscale = self.net(x).chunk(2, dim=1)
        if self.constrain_scale:
            logscale = logscale.tanh()
        z = (x - mean) * torch.exp(-logscale.clamp(min=-30.0, max=30.0))
        log_det = -logscale.sum(-1)
        return z, log_det

#--------------------------------------------------------------------------------------------------

class BatchNormTransform1d(_Transform):
    r"""
    1-dimensional Batch Normalization layer for stabilizing deep normalizing flows. It was
    first introduced in `Density Estimation Using Real NVP <https://arxiv.org/pdf/1605.08803.pdf>`_
    (Dinh et al., 2017).
    """

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        """
        Initializes a new batch normalization layer for one-dimensional vectors of the given
        dimension.

        Parameters
        ----------
        dim: int
            The dimension of the inputs.
        eps: float, default: 1e-5
            A small value added in the denominator for numerical stability.
        momentum: float, default: 0.1
            Value used for calculating running average statistics.
        """
        super().__init__(dim)

        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.empty(dim))
        self.beta = nn.Parameter(torch.empty(dim))

        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.ones(dim))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets this module's parameters.
        """
        nninit.zeros_(self.log_gamma) # equal to `init.ones_(self.gamma)`
        nninit.zeros_(self.beta)

    def forward(self, z):
        """
        Transforms the given input.

        Note
        ----
        During testing, inputs that highly differ from the inputs seen during testing, this module
        is generally prone to outputting non-finite float values. In that case, these inputs are
        considered to be "impossible" to observe: the transformed output is set to all zeros and
        the log-determinant is set to :code:`-inf`.

        Parameters
        ----------
        z: torch.Tensor [N, D]
            The given input (batch size N, dimensionality D).

        Returns
        -------
        torch.Tensor [N, D]
            The transformed input.
        torch.Tensor [N]
            The log-determinants of the Jacobian evaluated at z.
        """
        batch_size = z.size(0)

        if self.training:
            mean = z.mean(0)
            var = z.var(0, unbiased=True)

            # Use the .data property to prevent gradients from accumulating in the running stats
            self.running_mean.mul_(self.momentum).add_(mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(var.data * (1 - self.momentum))
        else:
            mean = self.running_mean
            var = self.running_var

        # normalize input
        x = (z - mean) / (var + self.eps).sqrt()
        out = x * self.log_gamma.exp() + self.beta

        # compute log-determinant
        log_det = self.log_gamma - 0.5 * (var + self.eps).log()
        # do repeat instead of expand to allow fixing the log_det below
        log_det = log_det.sum(-1).repeat(batch_size)

        # Fix an error where outputs are completely out of range during evaluation
        if not self.training:
            # Find all output rows where at least one value is not finite
            rows = (~torch.isfinite(out)).sum(1) > 0
            # Fill these rows with 0 and set the log-determinant to -inf to indicate that they have
            # a density of exactly 0
            out[rows] = (0)
            log_det[rows] = float('-inf')

        return out, log_det

#--------------------------------------------------------------------------------------------------

class LeakyReLUTransform(_Transform):
    """
    LeakyReLU non-linearity to be used for Normalizing Flows.
    """

    def __init__(self, negative_slope=0.01):
        """
        Initializes a new LeakyReLU transform.

        Parameters
        ----------
        negative_slope: float, default: 0.01
            The multiplier for negative values.
        """
        super().__init__(None)
        self.negative_slope = negative_slope
        self.log_det_factor = math.log(self.negative_slope)

    def forward(self, z):
        """
        Transforms the given input.

        Parameters
        ----------
        z: torch.Tensor [N, D]
            The given input (batch size N, dimensionality D).

        Returns
        -------
        torch.Tensor [N, D]
            The transformed input.
        torch.Tensor [N]
            The log-determinants of the Jacobian evaluated at z.
        """
        condition = z >= 0
        out = torch.where(condition, z, self.negative_slope * z)

        log_det_gte_0 = torch.zeros_like(z)
        log_det_lt_0 = torch.ones_like(z) * self.log_det_factor
        log_det_z = torch.where(condition, log_det_gte_0, log_det_lt_0)
        log_det = log_det_z.sum(-1)

        return out, log_det


class PReLUTransform(_Transform):
    """
    Parametric ReLU non-linearity to be used for Normalizing Flows. Compared to the standard PReLU,
    this implementation does not allow negative slopes.
    """

    def __init__(self, num_parameters=1, init=0.25, minimum=0.01):
        """
        Initializes a new parametric ReLU transform.

        Parameters
        ----------
        num_parameters: int, default: 1
            The number of parameters to use. Either 1 or the dimension of the normalizing flow.
            In the latter case, there exists one alpha value per dimension.
        init: float, default: 0.25
            The initial value for the parameter(s). Must be positive.
        minimum: float, default: 0.01
            The minimum attainable alpha value. Must be positive.
        """
        assert init > 0, "initial value must be positive"
        assert minimum > 0, "minimum value must be positive"

        super().__init__(None)

        self.init = init
        self.minimum = minimum
        self.weight_prime = nn.Parameter(torch.empty(num_parameters))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets this module's parameters.
        """
        nninit.constant_(self.weight_prime, self.init)

    def forward(self, z):
        """
        Transforms the given input.

        Parameters
        ----------
        z: torch.Tensor [N, D]
            The given input (batch size N, dimensionality D).

        Returns
        -------
        torch.Tensor [N, D]
            The transformed input.
        torch.Tensor [N]
            The log-determinants of the Jacobian evaluated at z.
        """
        condition = z >= 0
        weight = F.softplus(self.weight_prime) + self.minimum
        out = torch.where(condition, z, weight * z)

        log_det_gte_0 = torch.zeros_like(z)
        log_det_lt_0 = torch.ones_like(z) * weight.log()
        log_det_z = torch.where(condition, log_det_gte_0, log_det_lt_0)
        log_det = log_det_z.sum(-1)

        return out, log_det

    def __repr__(self):
        if self.weight_prime.numel() > 1:
            return f'{self.__class__.__name__}(dim={self.weight_prime.numel()})'
        alpha = F.softplus(self.weight_prime) + self.minimum
        return f'{self.__class__.__name__}(alpha={alpha.item():.2f})'

#--------------------------------------------------------------------------------------------------

class FlipTransform1d(_Transform):
    """
    Simple transform to flip the input. Required for stacking coupling layers and masked
    autoregressive transforms.
    """

    def __init__(self):
        super().__init__(None)

    def forward(self, z):
        """
        Flips the input along the second dimension.

        Parameters
        ----------
        z: torch.Tensor [N, D]
            The given input (batch size N, dimensionality D).

        Returns
        -------
        torch.Tensor [N, D]
            The flipped input.
        torch.Tensor [N]
            The log-determinants (zero).
        """
        return z.flip(-1), torch.zeros(z.size(0), dtype=z.dtype, device=z.device)
