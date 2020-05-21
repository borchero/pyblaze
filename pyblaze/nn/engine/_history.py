import json
import time
from collections import defaultdict
from ..callbacks import TrainingCallback

class History(TrainingCallback):
    """
    This class summarizes metrics obtained during training of a model. This class should never be
    initialized outside of this framework.
    """

    def __init__(self):
        """
        Initializes a new history object. When initializing, it sets up this object for logging
        against it.
        """
        self.start_time = None
        self.end_time = None
        self.metrics = defaultdict(list)

    ################################################################################
    ### CALLBACK METHODS
    ################################################################################
    def before_training(self, model, num_epochs):
        self.start_time = time.time()

    def after_training(self):
        self.end_time = time.time()

    def after_batch(self, metrics):
        self._log_metrics(metrics, prefix='batch_')

    def after_epoch(self, metrics):
        self._log_metrics(metrics)

    def _log_metrics(self, metrics, prefix=''):
        t = time.time()
        if isinstance(metrics, dict):
            # If metrics are dict, append to their names
            for key, value in metrics.items():
                self.metrics[f'{prefix}{key}'].append({'ts': t, 'value': value})
        elif isinstance(metrics, (list, tuple)):
            # If they are tuple, append to the loss plus their index
            for i, value in enumerate(metrics):
                self.metrics[f'{prefix}loss_{i}'].append({'ts': t, 'value': value})
        else:
            # Else, just call it loss
            self.metrics[f'{prefix}loss'].append({'ts': t, 'value': metrics})

    ################################################################################
    ### LOADING AND SAVING
    ################################################################################
    @classmethod
    def load(cls, file):
        """
        Loads the history object from the specified file.

        Parameters
        ----------
        file: str
            The JSON file to load the history object from.
        """
        with open(file, 'r') as f:
            loaded = json.load(f)
        return History(**loaded)

    def save(self, file):
        """
        Saves the history object to the specified file as JSON.

        Parameters
        ----------
        file: str
            The file (with .json extension) to save the history to.
        """
        obj = {'duration': self.duration, 'metrics': self._metrics}
        with open(file, 'w+') as f:
            json.dump(obj, f, indent=4, sort_keys=True)

    ################################################################################
    ### PUBLIC METHODS
    ################################################################################
    @property
    def keys(self):
        """
        Returns the available keys that this history object provides. The keys are returned as
        sorted list.
        """
        return sorted(self.metrics.keys())

    @property
    def duration(self):
        """
        Returns the duration that this history object recorded for the duration of the training.
        """
        return self.end_time - self.start_time

    def inspect(self, metric):
        """
        Returns all the datapoints recorded for the specified metric along with their timestamps.
        If the timestamps are not required, the metric may also be accessed by simple dot notation
        on the history object.

        Parameters
        ----------
        metric: str
            The name of the metric.

        Returns
        -------
        list of object
            The values recorded.
        list of float
            The timestamps at which the values were recorded. Has the same length as the other list.
        """
        if metric not in self.metrics:
            raise ValueError(f"Metric '{metric}' was not recorded.")

        values = [entry['value'] for entry in self.metrics[metric]]
        timestamps = [entry['ts'] for entry in self.metrics[metric]]
        return values, timestamps

    def __getattr__(self, name):
        return self.inspect(name)[0]

    def __str__(self):
        return f'History<{self.keys}>'

    def __repr__(self):
        return str(self)
