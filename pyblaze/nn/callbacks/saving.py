import torch
import torch.jit as jit
from pyblaze.utils.stdio import ensure_valid_directories
from .base import TrainingCallback

class ModelSaverCallback(TrainingCallback):
    """
    The callback stores the trained model after every epoch with a unique name per epoch. If the
    model uses the `pyblaze.nn.Configurable` mixin, its config and state dict are stored after
    every epoch, otherwise only its state dict.
    """

    def __init__(self, directory, file_template='model_epoch-{}'):
        """
        Initializes a new ModelSaverCallback.

        Parameters
        ----------
        directory: str
            The directory where the models should be stored.
        file_template: str, default: 'model_epoch_{}'
            A file template that can be formatted with a single integer, i.e. the epoch.
        """
        self.file_template = f'{directory}/{file_template}'
        self.model = None
        self.epoch = None

    def before_training(self, model, num_epochs):
        self.model = model

    def before_epoch(self, current, num_iterations):
        self.epoch = current

    def after_epoch(self, metrics):
        file = self.file_template.format(self.epoch)
        ensure_valid_directories(file)
        if isinstance(self.model, jit.ScriptModule):
            self.model.save(f'{file}.jit')
        else:
            torch.save(self.model.state_dict(), f'{file}.pt')
        self.epoch = None

    def after_training(self):
        self.model = None
