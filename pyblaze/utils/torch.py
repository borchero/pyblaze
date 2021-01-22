import torch
from torch.nn.utils.rnn import PackedSequence

def gpu_device(gpu):
    """
    Returns a device based on the passed parameters.

    Parameters
    ----------
    gpu: bool or int
        If int, the returned device is the GPU with the specified ID. If False, the returned device
        is the CPU, if True, the returned device is given as the GPU with the highest amount of
        free memory.

    Returns
    -------
    torch.device
        A PyTorch device.
    """
    if isinstance(gpu, bool) and gpu:
        assert torch.cuda.is_available()
        return torch.device('cuda', 0)

    if isinstance(gpu, bool):
        return torch.device('cpu')

    assert gpu < torch.cuda.device_count()
    return torch.device('cuda', gpu)


def to_device(device, *args):
    """
    Passes the given tensors to the specified device.

    Parameters
    ----------
    device: torch.device
        The device to pass tensors to.
    args: varargs of (torch.Tensor or list of torch.Tensor)
        The tensors to pass to the specified device. Tensors may be given in lists such that a
        single variable can easily be assigned to a list of tensors upon function return. The given
        tensors may also be None, then None is returned for that tensor.

    Returns
    -------
    list of (torch.Tensor or list of torch.Tensor)
        The given tensors passed to the specified device. If tensors were passed as lists, they are
        also returned as lists.
    """
    def _to_device(x):
        if isinstance(x, PackedSequence):
            return x.to(device, non_blocking=True)
        if x.is_sparse or x.is_contiguous():
            return x.to(device, non_blocking=True)
        return x.contiguous().to(device, non_blocking=True)

    return _recursive_apply('to', _to_device, *args)


def share_memory(*args):
    """
    Shares the memory of the given tensors.

    Parameters
    ----------
    args: varargs of (torch.Tensor or list of torch.Tensor)
        The tensors to share the memory.

    Returns
    -------
    list of (torch.Tensor or list of torch.Tensor)
        The tensors in shared memory.
    """
    return _recursive_apply('share_memory_', lambda x: x.share_memory_(), *args)


def pin_memory(*args):
    """
    Pins the memory of the given tensors.

    Parameters
    ----------
    args: varargs of (torch.Tensor or list of torch.Tensor)
        The tensors to pin the memory.

    Returns
    -------
    list of (torch.Tensor or list of torch.Tensor)
        The tensors with pinned memory.
    """
    return _recursive_apply('pin_memory', lambda x: x.pin_memory(), *args)


def _recursive_apply(attribute, function, *args):
    if len(args) > 1:
        return [_recursive_apply(attribute, function, t) for t in args]
    if len(args) == 1 and isinstance(args[0], list):
        return [_recursive_apply(attribute, function, t) for t in args[0]]
    if len(args) == 1 and isinstance(args[0], tuple) and not isinstance(args[0], PackedSequence):
        return tuple(_recursive_apply(attribute, function, t) for t in args[0])
    if len(args) == 1 and isinstance(args[0], dict):
        return {k: _recursive_apply(attribute, function, v) for k, v in args[0].items()}
    if len(args) == 1 and hasattr(args[0], attribute):
        return function(args[0])
    if len(args) == 1 and args[0] is not None:
        return args[0]
    return None
