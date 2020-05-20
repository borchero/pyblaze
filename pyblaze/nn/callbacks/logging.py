from pyblaze.utils.stdio import ProgressBar
from .base import TrainingCallback, PredictionCallback

class EpochProgressLogger(TrainingCallback):
    """
    Logs the training progress. It does only consider epochs (to plot the progress of each batch
    within an epoch, use `BatchProgressLogger`).
    """

    def __init__(self, file=None):
        """
        Initializes a new progress printer for epochs.

        Parameters
        ----------
        file: str, default: None
            If given, the progress is not written to the command line, but to a file instead. Might
            be especially useful when multiple processes perform training simultaneously.
        """
        self.progress = None
        self.file = file

    def before_training(self, model, num_epochs):
        self.progress = ProgressBar(num_epochs, self.file)
        self.progress.start()

    def after_epoch(self, metrics):
        self.progress.step()

    def after_training(self):
        self.progress.finish()
        self.progress = None


class BatchProgressLogger(TrainingCallback):
    """
    Logs the training progress. It does only consider batches (to plot the progress of the entire
    training, use `EpochProgressLogger`).
    """

    def __init__(self, file=None):
        """
        Initializes a new progress logger for batches.

        Parameters
        ----------
        file: str, default: None
            If given, the progress is not written to the command line, but to a file instead. Might
            be especially useful when multiple processes perform training simultaneously.
        """
        self.num_epochs = None
        self.progress = None
        self.file = file

    def before_training(self, model, num_epochs):
        self.num_epochs = num_epochs

    def before_epoch(self, current, num_iterations):
        print(f"Epoch {current+1}/{self.num_epochs}:")
        self.progress = ProgressBar(num_iterations, self.file)
        self.progress.start()

    def after_batch(self, metrics):
        self.progress.step()

    def after_epoch(self, metrics):
        self.progress.finish(metrics)
        self.progress = None

    def after_training(self):
        self.num_epochs = None


class PredictionProgressLogger(PredictionCallback):
    """
    Logs the prediction progress.
    """

    def __init__(self, file=None):
        """
        Initializes a new progress printer for predictions.

        Parameters
        ----------
        file: str, default: None
            If given, the progress is not written to the command line, but to a file instead. Might
            be especially useful when multiple processes perform training simultaneously.
        """
        self.progress = None
        self.file = file

    def before_predictions(self, model, num_iterations):
        self.progress = ProgressBar(num_iterations, self.file)
        self.progress.start()

    def after_batch(self, *args):
        self.progress.step()

    def after_predictions(self):
        self.progress.finish()
        self.progress = None
