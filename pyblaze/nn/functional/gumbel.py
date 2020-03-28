import torch
import torch.jit as jit
import torch.nn.functional as F

@jit.script
def gumbel_softmax(logits: torch.Tensor,
                   tau: float = 1,
                   hard: bool = False,
                   eps: float = 1e-10,
                   dim: int = -1):
    """
    Numerically stable version of PyTorch's builtin gumbel softmax.

    Parameters
    ----------
    logits: torch.Tensor
        The values fed into the gumbel softmax.
    tau: float, default: 1
        Temperature parameter for the Gumbel distribution.
    hard: bool, default: False
        Whether to obtain a one-hot output.
    eps: float, default: 1e-10
        Constant for numerical stability.
    dim: int, default: -1
        The dimension over which to apply the softmax.
    """
    uniform_sample = torch.empty_like(logits).uniform_(0.0, 1.0)
    gumbel_sample = -torch.log(eps - torch.log(uniform_sample + eps))

    y_soft = F.softmax((logits + gumbel_sample) / tau, dim=dim)

    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    return y_soft
