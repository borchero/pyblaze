import os
from .base import TrainingCallback, CallbackException

class SynchronizationCallback(TrainingCallback):
    """
    Synchronizer callback which can be used within a parallel environment to stop and resume
    training at specific points.
    """

    def __init__(self, push_queue, pull_queue):
        """
        Initializes a new synchronizer.

        Parameters
        ----------
        push_queue: torch.multiprocessing.Queue
            The queue onto which the parallel environment pushes whether the
            current epoch should be trained. If `False` is passed, training
            is stopped.
        pull_queue: torch.multiprocessing.Queue
            The queue onto which the callback pushes `True` to indicate that the epoch is finished.
            Upon end of training, it pushes `False`.
        """
        self.push_queue = push_queue
        self.pull_queue = pull_queue

    def before_epoch(self, current, num_iterations):
        cont = self.push_queue.get()
        if not cont:
            raise CallbackException(
                f'Process {os.getpid()} is told to finish training.'
            )

    def after_epoch(self, metrics):
        self.pull_queue.cancel_join_thread()
        self.pull_queue.put(True)

    def after_training(self):
        self.pull_queue.cancel_join_thread()
        self.pull_queue.put(False)


class ModelSharingCallback(TrainingCallback):
    """
    The model sharing callback should be used in a multiprocessing context where a model is shared
    in memory, but workers are operating on a GPU and cannot share the model directly. This
    callback ensures that a trainer always reads updated from other workers at the beginning of
    every batch.
    """

    def __init__(self, shared_model):
        """
        Initializes a new model sharing callback.

        Parameters
        ----------
        shared_model: torch.nn.Module
            The model that should be shared. Must have already been moved to shared memory.
        """
        self.shared_model = shared_model
        self.trained_model = None

    def before_training(self, model, num_epochs):
        # Store the model being trained
        self.trained_model = model

    def before_epoch(self, current, num_iterations):
        # Load parameters from shared model
        weights = self.shared_model.state_dict()
        self.trained_model.load_state_dict(weights)

    def after_training(self):
        self.trained_model = None
