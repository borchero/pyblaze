import torch

def logmeanexp(x, dim, keepdim=False, allow_inf=False):
    """
    Returns the log of the exponentiated means over the given dimension(s) in a numerically stable
    way. This is similar to :code:`torch.logsumexp`, but computes the mean instead of the sum in
    the exp-space.

    Parameters
    ----------
    x: torch.Tensor
        The input tensor.
    dim: int or list of int
        The dimension(s) to perform the computation over.
    keepdim: bool, default: False
        Whether to retain the dimension of the input.
    allow_inf: bool, default: False
        Whether to allow infinities in the input. Cannot be set if gradients are required.
    """
    x_max = x.amax(dim, keepdim=True)
    if allow_inf:
        # If we don't require gradients, we can allow infinities in the input
        x_max[torch.isinf(x_max)] = 0
    z = (x - x_max).exp()
    r = z.mean(dim, keepdim=True).log() + x_max
    if keepdim:
        return r
    return r.squeeze(dim)
