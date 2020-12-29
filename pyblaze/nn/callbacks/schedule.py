from .base import TrainingCallback, ValueTrainingCallback

class LearningRateScheduler(TrainingCallback):
    """
    The learning rate scheduler may be used with a PyTorch learning rate scheduler. The callback is
    automatically triggered after the end of every iteration or epoch.
    """

    def __init__(self, scheduler, metric=None, after_batch=False):
        """
        Initializes a new learning rate scheduler for the given PyTorch scheduler.

        Parameters
        ----------
        scheduler: torch.optim.lr_scheduler
            The PyTorch scheduler.
        metric: str, default: None
            The metric to pass to the scheduler, e.g. useful for reducing the learning rate as the
            validation loss pleateaus. Typically, it should only be used with :code:`after_batch`
            set to `False`.
        after_batch: bool, default: False
            Whether to call the scheduler after every batch or after every epoch.
        """
        self.exec_after_batch = after_batch
        self.metric = metric
        self.scheduler = scheduler

    def after_batch(self, metrics):
        if self.exec_after_batch:
            self._exec(metrics)

    def after_epoch(self, metrics):
        if not self.exec_after_batch:
            self._exec(metrics)

    def _exec(self, metrics):
        if self.metric is not None:
            self.scheduler.step(metrics[self.metric])
        else:
            self.scheduler.step()


class ParameterScheduler(ValueTrainingCallback):
    """
    The parameter scheduler is able to change the value of a variable over the course of the
    training.
    """

    def __init__(self, initial, schedule, *args, **kwargs):
        r"""
        Initalizes a new scheduler for the given parameter.

        Parameters
        ----------
        initial: object
            The initial value fo the parameter which should be modified over the course of the
            training.
        schedule: func (object, int, int, \**kwargs) -> object
            Function which should return the value of the parameter based on the current value of
            the parameter, the current epoch, and the iteration within the epoch. The function is
            called after every iteration (i.e. batch). It is further passed the arguments given to
            this initializer.
        args: variadic argument
            Additional arguments passed to the :code:`schedule` function.
        kwargs: keyword arguments
            Additional keyword arguments passed to the :code:`schedule` function.
        """
        self.parameter = initial
        self.schedule = schedule
        self.args = args
        self.kwargs = kwargs
        self.epoch = None
        self.iteration = None

    def read(self):
        return self.parameter

    def before_training(self, model, num_epochs):
        self.iteration = 0

    def before_epoch(self, current, num_iterations):
        self.epoch = current

    def after_batch(self, metrics):
        self.iteration += 1
        self._update()

    def after_epoch(self, metrics):
        self._update()

    def after_training(self):
        self.epoch = None
        self.iteration = None

    def _update(self):
        self.parameter = self.schedule(
            self.parameter, self.epoch, self.iteration, *self.args, **self.kwargs
        )
