from .base import Engine
from .utils import forward

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
        by that.
    """

    def __init__(self, model):
        super().__init__(model)
        self._grad_accum_cache = {}

    def before_epoch(self, current, num_iterations):
        self._grad_accum_cache['num_iterations'] = num_iterations
        self._grad_accum_cache['current_iteration'] = 0

    # pylint: disable=unused-argument
    def after_batch(self, *args):
        if 'current_iteration' in self._grad_accum_cache:
            self._grad_accum_cache['current_iteration'] += 1

    def after_epoch(self, metrics):
        self._grad_accum_cache = {}

    def train_batch(self, data, optimizer=None, loss=None, gradient_accumulation_steps=1):
        it = self._grad_accum_cache['current_iteration']
        if it == 0:
            optimizer.zero_grad()

        loss_func = loss

        x, y_true = data
        y_pred = forward(self.model, x)
        loss = loss_func(y_pred, y_true) / gradient_accumulation_steps

        loss.backward()

        it_is_last = it == self._grad_accum_cache['num_iterations'] - 1
        if it % gradient_accumulation_steps == 0 or it_is_last:
            optimizer.step()
            optimizer.zero_grad()

        return loss.item()

    def eval_batch(self, data):
        x, y_true = data
        y_pred = forward(self.model, x)
        return y_pred, y_true
