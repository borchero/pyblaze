import torch.nn as nn

class DataParallel(nn.DataParallel):
    """
    Replacement for PyTorch's default `nn.DataParallel` class. It enables accessing the
    attributes of the underlying module. If the attribute being accessed is a module itself, it is
    also returned being wrapped as DataParallel. Call `.module` to access the raw model.
    """

    def __init__(self, model, device_ids=None, output_device=None, dim=0):
        super().__init__(model, device_ids, output_device, dim)
        self._cache = {}

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            value = getattr(self.module, name)
            if isinstance(value, nn.Module):
                if value not in self._cache:
                    self._cache[value] = DataParallel(
                        value, device_ids=self.device_ids,
                        output_device=self.output_device,
                        dim=self.dim
                    )
                return self._cache[value]
            return value
