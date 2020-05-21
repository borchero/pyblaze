from .base import Engine
from ._utils import forward

class LikelihoodEngine(Engine):
    """
    Engine to be used for training a model that ought to fit some data without targets. The
    likelihood of the data should be maximized (i.e. the sum of negative log-likelihoods (NLLs)
    should be minimized). The supplied model must return a valid per-element log-likelihood from its
    :code:`forward` method.

    The engine requires data to be available in the following format:

    Parameters
    ----------
    x: object
        The input to the model to evaluate the NLL for.

    The :meth:`train` method allows for the following keyword arguments:

    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        The optimizer to use for the model.
    """

    def train_batch(self, data, optimizer=None):
        optimizer.zero_grad()

        log_likeli = forward(self.model, data)
        loss = -log_likeli.mean()
        loss.backward()

        optimizer.step()

        return {'nll': loss.item()}

    def eval_batch(self, data):
        return forward(self.model, data)
