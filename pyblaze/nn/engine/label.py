from .base import Engine
from ._utils import forward

class LabelEngine(Engine):
    """
    Engine to be used in a supervised learning setting where labels for the data are available.


    The engine requires data to be available in the following format:

    Parameters
    ----------
    x: object
        The input to the model. The type depends on the model. Usually, however, it is a tensor of
        shape [N, ...] for batch size N. This value is always required.
    y: object
        The labels corresponding to the input. The type depends on the model. Usually, it is set to
        the class indices (for classification) or the target values (for regression) and therefore
        given as tensor of shape [N]. This value must not be given if :meth:`predict` is called.


    For the :meth:`train` method allows for the following keyword arguments:

    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        The optimizer to use for the model.
    loss: torch.nn.Module
        The loss function to use. It receives two inputs: the output of the model and the target
        values. Usually, they are given as `(torch.Tensor [N], torch.Tensor [N])` for batch size N.
    gradient_accumulation_steps: int, default: 1
        The number of batches which should be used for a single update step. Gradient accumulation
        can be useful if the GPU can only fit a small batch size but model convergence is hindered
        by that. The number of gradient accumulation steps may not be changed within an epoch.
    """

    def __init__(self, model):
        super().__init__(model)

        self.num_it = None
        self.current_it = None

    def before_epoch(self, current, num_iterations):
        self.num_it = num_iterations
        self.current_it = 0

    # pylint: disable=unused-argument
    def after_batch(self, *args):
        if self.current_it is not None:
            self.current_it += 1

    def after_epoch(self, metrics):
        self.num_it = None
        self.current_it = None

    def train_batch(self, data, optimizer=None, loss=None, gradient_accumulation_steps=1):
        if self.current_it == 0:
            optimizer.zero_grad()

        loss_func = loss

        x, y_true = data
        y_pred = forward(self.model, x)
        loss = loss_func(y_pred, y_true) / gradient_accumulation_steps

        loss.backward()

        last_it = self.current_it == self.num_it - 1
        if self.current_it % gradient_accumulation_steps == 0 or last_it:
            optimizer.step()
            optimizer.zero_grad()

        return loss.item()

    def eval_batch(self, data):
        x, y_true = data
        y_pred = forward(self.model, x)
        return y_pred, y_true
