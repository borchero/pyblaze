import json
import numpy as np
from pyblaze.utils.stdlib import flatten

class History:
    """
    This class summarizes metrics obtained during training of a model.
    """

    @staticmethod
    def load(file):
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

    @staticmethod
    def cat(hist1, hist2):
        """
        Concatenates the two history objects.

        Parameters
        ----------
        hist1: pyblaze.nn.History
            The first history.
        hist2: pyblaze.nn.History
            The second history.

        Returns
        -------
        pyblaze.nn.History
            The concatenated history object.
        """
        # pylint: disable=protected-access
        return History(
            hist1.duration + hist2.duration,
            hist1._metrics + hist2._metrics
        )

    def __init__(self, duration, metrics):
        """
        Initializes a new history object.

        Parameters
        ----------
        metrics: list of dict
            The metrics from each training step.
        """
        self.duration = duration
        self._metrics = metrics

    @property
    def keys(self):
        """
        Returns the available keys that this history object provides.

        Returns
        -------
        list of str
            The keys.
        """
        return sorted(self._metrics[0].keys())

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

    def __len__(self):
        return len(self._metrics)

    def __getattr__(self, name):
        if name == '_metrics':
            raise AttributeError()
        if len(self._metrics) == 0:
            raise AttributeError(
                f'Length of history is 0. Metric {name} cannot be accessed.'
            )
        if name in self._metrics[0]:
            result = [m[name] for m in self._metrics]
            if name == 'micro_train_loss':
                return flatten(result)
            return result
        raise AttributeError(f'Metric {name} does not exist.')


class Evaluation:
    """
    This class summarizes metrics obtained when evaluating a model.
    """

    @staticmethod
    def merge(eval1, eval2, *eval_other):
        """
        Merges multiple evaluations into one.

        Parameters
        ----------
        eval1: torch.nn.training.wrappers.Evaluation
            The first evaluation.
        eval2: torch.nn.training.wrappers.Evaluation
            The second evaluation.
        eval_other: varargs of torch.nn.training.wrappers.Evaluation
            Additional evaluations.

        Returns
        -------
        pyblaze.nn.Evaluation
            The merged evaluation.
        """
        # pylint: disable=protected-access
        result = Evaluation({})
        metrics = {}
        for e in [eval1, eval2] + list(eval_other):
            for k, v in e._metrics.items():
                metrics[k] = v
        result._metrics = metrics
        return result

    def __init__(self, metrics, weights=None):
        """
        Initializes a new evaluation wrapper from the given metrics.

        Parameters
        ----------
        metrics: dict of str -> (list of float or float)
            The metrics computed. The keys define the metric names, the values provide the metrics
            computed over all batches.
        weights: list of int, default: None
            The length of the batches. Should be supplied if the batches for which the metrics have
            been computed have differing sizes. If so, this list must contain the sizes of all
            batches.
        """
        result = {}
        for metric, values in metrics.items():
            if isinstance(values, float):
                average = values
            elif weights is None or len(weights) == 0:
                average = np.mean(values)
            else:
                weights = np.array(weights)
                values = np.array(values)
                average = weights.dot(values) / np.sum(weights)
            result[metric] = average
        self._metrics = result

    @property
    def keys(self):
        """
        Returns the available keys for this evaluation object.

        Returns
        -------
        list of str
            The keys.
        """
        return sorted(self._metrics.keys())

    def to_dict(self):
        """
        Returns the evaluation object as plain dictionary.

        Returns
        -------
        dict
            The dictionary.
        """
        return {k: v for k, v in self._metrics.items() if not k.startswith('_')}

    def with_prefix(self, prefix):
        """
        Initializes a new evaluation wrapper where all metrics are prefixed with the given prefix.
        Useful when merging evaluations.

        Parameters
        ----------
        prefix: str
            The prefix to use.

        Returns
        -------
        pyblaze.nn.Evaluation
            The resulting evaluation.
        """
        # pylint: disable=protected-access
        result = Evaluation({})
        result._metrics = {f'{prefix}{k}': v for k, v in self._metrics.items()}
        return result

    def __contains__(self, item):
        return item in self._metrics

    def __getitem__(self, item):
        return self._metrics[item]

    def __getattr__(self, name):
        if name == '_metrics':
            raise AttributeError()
        try:
            return self._metrics[name]
        except KeyError:
            class_name = self.__class__.__name__
            raise AttributeError(
                f'{class_name} does not provide metric {name}.'
            )
