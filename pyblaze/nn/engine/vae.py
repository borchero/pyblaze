from .base import Engine
from .utils import forward

class VAEEngine(Engine):
    """
    Engine to be used for training a variational autoencoder.

    The autoencoder must be supplied as a single model with :code:`encoder` and :code:`decoder`
    attributes. The decoder must output a 3-tuple: the reconstructed input, as well as the mean and
    the log-variance as outputted by the encoder.

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

    def train_batch(self, data, optimizer=None, loss=None):
        optimizer.zero_grad()

        x_out, mu, logvar = forward(self.model, data)
        loss_val = loss(x_out, data, mu, logvar)
        loss_val.backward()

        optimizer.step()

        return loss.item()

    def eval_batch(self, data):
        x_out, _, _ = forward(self.model, data)
        return x_out, data
