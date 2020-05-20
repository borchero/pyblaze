from abc import ABC, abstractmethod

class CallbackException(Exception):
    """
    Exception raised by callbacks to stop the training procedure.
    """

    def __init__(self, message, verbose=False):
        super().__init__(message)
        self.verbose = verbose

    def print(self):
        """
        Prints the message of this exception if it is verbose. Otherwise it is a no-op.
        """
        if self.verbose:
            print(self)


class TrainingCallback(ABC):
    """
    Abstract class to be subclassed by all training callbacks. These callbacks are passed to an
    engine which calls the implemented methods at specific points during training.
    """

    def before_training(self, model, num_epochs):
        """
        Method is called prior to the start of the training. This method must not raise exceptions.

        Parameters
        ----------
        model: torch.nn.Module
            The model which is trained.
        num_epochs: int
            The maximum number of epochs performed during training.
        """

    def before_epoch(self, current, num_iterations):
        """
        Method is called at the begining of every epoch during training. This method may raise
        exceptions if training should be stopped. Note, however, that stopping training at this
        stage is an advanced scenario.

        Parameters
        ----------
        current: int
            The index of the epoch that is about to start.
        num_iterations: int
            The expected number of iterations for the batch.
        """

    def after_batch(self, metrics):
        """
        Method is called at the end of a mini-batch. If the data is not partitioned into batches,
        this function is never called. The method may not raise exceptions.

        Parameters
        ----------
        metrics: float or tuple or dict
            The metrics obtained from the last mini-batch. This is the value that is returned from
            an engine's :meth:`train_batch` method.
        """

    def after_epoch(self, metrics):
        """
        Method is called at the end of every epoch during training. This method may raise
        exceptions if training should be stopped.

        Parameters
        ----------
        metrics: float or tuple or dict
            Metrics obtained after training this epoch.
        """

    def after_training(self):
        """
        Method is called upon end of the training procedure. The method may not raise exceptions.
        """


class ValueTrainingCallback(TrainingCallback, ABC):
    """
    A training callback with an additional method that can be used to obtain a dynamically
    changing value from the callback.
    """

    @abstractmethod
    def read(self):
        """
        Returns the value that this training callback stores.
        """


class PredictionCallback(ABC):
    """
    Abstract class to be subclassed by all prediction callbacks. These callbacks are passed to an
    engine which calls the implemented methods at specific points during inference.
    """

    def before_predictions(self, model, num_iterations):
        """
        Called before prediction making starts.

        Parameters
        ----------
        model: torch.nn.Module
            The model which is used to make predictions.
        num_iterations: int
            The number of iterations/batches performed for prediction.
        """

    def after_batch(self, *args):
        """
        Called after prediction is done for one batch.

        Parameters
        ----------
        args: varargs
            Usually empty, just to be able to implement both TrainingCallback and
            PredictionCallback.
        """

    def after_predictions(self):
        """
        Called after all predictions have been made.
        """
