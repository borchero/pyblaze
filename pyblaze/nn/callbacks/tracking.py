from .base import TrainingCallback

class NeptuneTracker(TrainingCallback):
    """
    The Neptune tracker can be used to track experiments with https://neptune.ai. As soon as metrics
    are available they are logged to the experiment that this tracker is managing. It requires
    `neptune-client` to be installed.
    """

    def __init__(self, experiment):
        """
        Initializes a new tracker for the given neptune experiment.

        Parameters
        ----------
        experiment: neptune.experiments.Experiment
            The experiment to log for.
        """
        self.experiment = experiment

    def after_batch(self, train_loss):
        if isinstance(train_loss, (list, tuple)):
            for i, val in enumerate(train_loss):
                self.experiment.log_metric(f'batch_train_loss_{i}', val)
        elif isinstance(train_loss, dict):
            for k, val in train_loss.items():
                self.experiment.log_metric(f'batch_train_loss_{k}', val)
        else:
            self.experiment.log_metric('batch_train_loss', train_loss)

    def after_epoch(self, metrics):
        for k, v in metrics.to_dict().items():
            self.experiment.log_metric(k, v)
