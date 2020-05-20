from .base import Engine
from ._utils import forward

class VAEEngine(Engine):
    """
    Engine to be used for training a variational autoencoder.

    The autoencoder must be supplied as a single model. The decoder must output a 3-tuple: the
    reconstructed input, as well as the mean and the log-variance as outputted by the encoder.

    The engine requires data to be available in the following format:

    Parameters
    ----------
    x: object
        The input to the encoder (as well as the target).

    The :meth:`train` method allows for thhe following keyword arguments:

    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        The optimizer to use for the model.
    loss: torch.nn.Module
        The loss function to use. It receives a 4-tuple of the decoder's output, the input, and the
        mean as well as log-variance outputted by the encoder :code:`(x_out, x_in, mu, logvar)`.
    """

    def __init__(self, model, ignore_target=False):
        """
        Initializes a new engine for training a VAE.

        Parameters
        ----------
        model: torch.nn.Module
            The VAE to train.
        ignore_target: bool, default: False
            When this value is set to `True`, the data passed to this engine is expected to yield
            class labels. They are discarded as a result.
        """
        super().__init__(model)
        self.ignore_target = ignore_target

    def train_batch(self, data, optimizer=None, loss=None):
        x = self._get_x(data)

        optimizer.zero_grad()

        x_pred, mu, logvar = forward(self.model, x)
        loss_val = loss(x_pred, x, mu, logvar)
        loss_val.backward()

        optimizer.step()

        return loss_val.item()

    def eval_batch(self, data):
        x = self._get_x(data)
        x_pred, mu, logvar = forward(self.model, x)
        return x_pred, x, mu, logvar

    def _get_x(self, data):
        if self.ignore_target:
            return data[0]
        return data
