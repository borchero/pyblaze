import copy
from .base import TrainingCallback, CallbackException

class EarlyStopping(TrainingCallback):
    """
    The early stopping callback watches a specified metric and interrupts training if the metric
    does not decrease for a specified number of epochs.
    """

    def __init__(self, metric='val_loss', patience=5, restore_best=False, minimize=True):
        """
        Initializes a new early stopping callback.

        Parameters
        ----------
        metric: str or list of str, default: 'val_loss'
            The metric to watch during training. If a list is given, the sum of the given metrics
            is considered.
        patience: int, default: 5
            The number of epochs that training still continues although the watched metric is
            greater than its smallest value during training.
        restore_best: bool, default: False
            Whether the model's parameter should be set to the parameters which showed the best
            performance in terms of the watched metric.
        minimize: bool, default: True
            Whether to minimize or maximize the given metric.
        """
        self.patience = patience
        self.epoch = None
        self.counter = None
        self.best_metric = None
        self.model = None
        self.state_dict = None
        self.restore_best = restore_best
        self.metric = metric
        self.minimize = minimize

    def before_training(self, model, num_epochs):
        self.model = model
        if self.restore_best:
            self.state_dict = copy.deepcopy(model.state_dict())
        self.epoch = 0
        self.counter = 0
        self.best_metric = float('inf') if self.minimize else -float('inf')

    def after_epoch(self, metrics):
        prev_epoch = self.epoch
        self.epoch += 1

        try:
            is_better = self._is_metric_better(metrics)
        except KeyError:
            if prev_epoch > 0:
                # In this case, we can ignore the key error and just skip -- the engine does not
                # perform evaluation on every iteration
                return

        if is_better:
            if self.restore_best:
                self.state_dict = copy.deepcopy(self.model.state_dict())
            self.counter = 0
            self.best_metric = self._current_metric(metrics)
        else:
            self.counter += 1
            if self.counter == self.patience:
                raise CallbackException(
                    f"Early stopping after epoch {self.epoch} (patience {self.patience}).",
                    verbose=True
                )

    def after_training(self):
        if self.restore_best and self.counter > 0:
            self.model.load_state_dict(self.state_dict)

    def _is_metric_better(self, metrics):
        if self.minimize:
            return self._current_metric(metrics) < self.best_metric
        return self._current_metric(metrics) > self.best_metric

    def _current_metric(self, metrics):
        if isinstance(self.metric, list):
            return sum(metrics[m] for m in self.metric)
        return metrics[self.metric]
