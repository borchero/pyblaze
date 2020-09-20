from .base import Engine
from ._utils import forward

class MLEEngine(Engine):
    """
    This engine can be used for all kinds of learning tasks where some kind of likelihood is
    maximized. This includes both of supervised and unsupervised learning. In general, this engine
    should be used whenever a model simply optimizes a loss function.


    The engine requires data to be available in the following format:

    Parameters
    ----------
    x: object
        The input to the model. The type depends on the model. Usually, however, it is a tensor of
        shape [N, ...] for batch size N. This value is always required.
    y: object
        The labels corresponding to the input. The type depends on the model. Usually, it is set to
        the class indices (for classification) or the target values (for regression) and therefore
        given as tensor of shape [N]. In case this value is given as tuple, all components are
        given to the loss function. This is e.g. useful when using a triplet loss. The targets
        might or might not be given for unsupervised learning tasks. In the latter case,
        :code:`expects_data_target` should be set to `False` in the engine's initializer. Targets
        must not be given if :meth:`predict` is called.


    For the :meth:`train` method allows for the following keyword arguments:

    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        The optimizer to use for the model.
    loss: torch.nn.Module
        The loss function to use. The number of inputs it receives depends on the model being
        trained and the kind of learning problem. In the most general form, the parameters are
        given as `(<all model outputs...>, <all targets...>, <all inputs...>)` where the model
        output is expected to be a tuple. The passing of targets and inputs to the loss function
        may be toggled via the :code:`uses_data_target` and :code:`uses_data_input` parameters of
        the engine's initializer. In the most common (and default) case, inputs are simply given as
        `(torch.Tensor [N], torch.Tensor [N])` for batch size N. The output of this function must
        always be a `torch.Tensor [1]`.
    gradient_accumulation_steps: int, default: 1
        The number of batches which should be used for a single update step. Gradient accumulation
        can be useful if the GPU can only fit a small batch size but model convergence is hindered
        by that. The number of gradient accumulation steps may not be changed within an epoch.
    kwargs: keyword arguments
        Additional keyword arguments passed to the :meth:`forward` method of the model. These
        keyword arguments may also be passed for the :meth:`eval_batch` method call.

    Note
    ----
    The same parameters that are given to the loss are also given to any metrics used with this
    engine.
    """

    def __init__(self, model, expects_data_target=True, uses_data_target=True,
                 uses_data_input=False):
        """
        Initializes a new MLE engine.

        Parameters
        ----------
        model: torch.nn.Module
            The model to train or evaluate.
        expects_data_target: bool, default: True
            Whether the data supplied to this engine is a tuple consisting of data and target or
            only contains the data. Setting this to `False` is never required for supervised
            learning.
        uses_data_target: bool, default: True
            Whether the loss function requires the target. Setting this to `False` although a data
            target is expected might make sense when the dataset comes with targets and it is
            easier to not preprocess it. This value is automatically set to `False` when no target
            is expected.
        uses_data_input: bool, default: True
            Whether the loss function requires the input to the model. This is e.g. useful for
            unsupervised learning tasks.
        """
        super().__init__(model)

        self.expects_data_target = expects_data_target
        self.uses_data_target = uses_data_target and expects_data_target
        self.uses_data_input = uses_data_input

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

    ################################################################################
    ### MAIN IMPLEMENTATION
    ################################################################################
    def train_batch(self, data, optimizer=None, loss=None, gradient_accumulation_steps=1, **kwargs):
        if self.current_it == 0:
            optimizer.zero_grad()

        # Get the actual data
        x, target = self._get_x_target(data)

        # Get the model output
        out = forward(self.model, x, **kwargs)

        # Apply the loss
        loss_input = self._merge_out_target_x(out, target, x)
        loss_val = forward(loss, loss_input) / gradient_accumulation_steps
        loss_val.backward()

        # Check if the optimizer should take a step
        last_it = self.current_it == self.num_it - 1
        if self.current_it % gradient_accumulation_steps == 0 or last_it:
            optimizer.step()
            optimizer.zero_grad()

        # Return the loss
        return loss_val.item()

    def eval_batch(self, data, **kwargs):
        x, target = self._get_x_target(data)
        out = forward(self.model, x, **kwargs)
        return self._merge_out_target_x(out, target, x)

    def predict_batch(self, data, **kwargs):
        x, _ = self._get_x_target(data)
        out = forward(self.model, x, **kwargs)
        return out

    ################################################################################
    ### PRIVATE
    ################################################################################
    def _get_x_target(self, data):
        if self.expects_data_target:
            return data
        return data, None

    def _merge_out_target_x(self, out, target, x):
        if not isinstance(out, (list, tuple)):
            out = (out,)
        out = tuple(out)

        if self.uses_data_target:
            if isinstance(target, (list, tuple)):
                out += tuple(target)
            else:
                out += (target,)

        if self.uses_data_input:
            if isinstance(x, (list, tuple)):
                out += tuple(x)
            else:
                out += (x,)

        return out
