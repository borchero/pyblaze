from torch.utils.tensorboard import SummaryWriter
from .base import TrainingCallback
try:
    import wandb
except ModuleNotFoundError:
    pass

class Tracker(TrainingCallback):
    """
    Abstract class for implementing a tracking callback using tracking frameworks which already
    include step-counters (e.g. for epochs) when logging metrics, e.g. NeptuneTracker or tracking
    with sacred.
    """

    def after_batch(self, train_loss):
        if isinstance(train_loss, (list, tuple)):
            for i, val in enumerate(train_loss):
                self.log_metric(f'batch_train_loss_{i}', val)
        elif isinstance(train_loss, dict):
            for k, val in train_loss.items():
                self.log_metric(f'batch_train_loss_{k}', val)
        else:
            self.log_metric('batch_train_loss', train_loss)

    def after_epoch(self, metrics):
        for k, v in metrics.items():
            self.log_metric(k, v)

    def log_metric(self, name, val):
        """
        Logs the given value for the specified metric.
        """
        raise NotImplementedError


class CounterTracker(TrainingCallback):
    """
    Abstract class for implementing a tracking callback for tracking frameworks which require
    passing the step-count (e.g. epoch) e.g. when tracking with tensorboard.
    """

    def __init__(self, writer):
        super().__init__()
        self.writer = writer
        self.batch_counter = None
        self.epoch_counter = None

    def before_training(self, model, num_epochs):
        self.batch_counter = 0
        self.epoch_counter = 0

    def after_batch(self, train_loss):
        if isinstance(train_loss, (list, tuple)):
            for i, val in enumerate(train_loss):
                self.log_metric(f'batch_train_loss_{i}', val, self.batch_counter)
        elif isinstance(train_loss, dict):
            for k, val in train_loss.items():
                self.log_metric(f'batch_train_loss_{k}', val, self.batch_counter)
        else:
            self.log_metric('batch_train_loss', train_loss, self.batch_counter)
        self.batch_counter += 1

    def after_epoch(self, metrics):
        for k, v in metrics.items():
            self.log_metric(k, v, self.epoch_counter)
        self.epoch_counter += 1

    def log_metric(self, name, val, step):
        """
        Logs the given value for the specified metric at the given step.
        """
        raise NotImplementedError


class NeptuneTracker(Tracker):
    """
    The Neptune tracker can be used to track experiments with https://neptune.ai. As soon as metrics
    are available they are logged to the experiment that this tracker is managing. It requires
    `neptune-client` to be installed.

    __init__
        Initializes a new tracker for the given neptune experiment.

        Parameters
        ----------
        experiment: neptune.experiments.Experiment
            The experiment to log for.
    """

    def __init__(self, experiment):
        self.experiment = experiment

    def log_metric(self, name, val):
        self.experiment.log_metric(name, val)


class SacredTracker(Tracker):
    """
    SimpleTracker which works together with Sacred. By using a NeptuneObserver, this
    tracker also allows for tracking the experiments with https://neptune.ai (see above)

    __init__
        Initializes a new tracker for the given sacred experiment.

        Parameters
        ----------
        experiment: sacred.Experiment
            The experiment to log for.
    """

    def __init__(self, experiment):
        self.experiment = experiment

    def log_metric(self, name, val):
        self.experiment.log_scalar(name, val)


class TensorboardTracker(CounterTracker):
    """
    The tensorboard tracker can be used to track experiments with tensorboard. The summary writer
    is available as `writer` property on the tracker.

    __init__
        Initializes a new Tensorboard tracker logging to the specified directory.

        Parameters
        ----------
        local_dir: str
            The directory to log to.
    """

    def __init__(self, local_dir):
        super().__init__(writer=SummaryWriter(local_dir))

    def log_metric(self, name, val, step):
        self.writer.add_scalar(name, val, step)


class WandbTracker(Tracker):
    """
    The wandb tracker allows tracking experiments with https://www.wandb.com/.
    """

    def log_metric(self, name, val):
        wandb.log({name: val})
