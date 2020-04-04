import os
import time
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.cuda as cuda
import pyblaze.multiprocessing as xmp
from pyblaze.nn.callbacks import TrainingCallback, PredictionCallback, CallbackException
import pyblaze.nn.utils as xnnu
from pyblaze.utils.torch import gpu_device, to_device
from pyblaze.utils.stdlib import flatten
from .wrappers import History, Evaluation

class BaseEngine(TrainingCallback, PredictionCallback, ABC):
    """
    A base class for training and evaluating models as well as making predictions. Generally, this
    class should be seen as _binding_ between a model and some data. Models should always be
    wrapped in an engine, both when being trained, and when performing inference/evaluation. The
    engine ensures that the model's environment is correct and prevents plenty of pitfalls.

    A concrete implementation of this class is tailored to a specific type of data (e.g.
    independent, identically distributed data samples) and/or model types (e.g. GANs).
    """

    #########################
    ### ENGINE PROPERTIES ###
    #########################
    def supports_multiple_gpus(self):
        """
        Returns whether the engine allows multiple GPUs to be used during training. By default, it
        returns `True`.

        Returns
        -------
        bool
            Whether multiple GPUs are allowed.
        """
        return True

    ######################
    ### INITIALIZATION ###
    ######################
    def __init__(self, model):
        """
        Initializes a new engine for a specified model.

        Parameters
        ----------
        model: torch.nn.Module
            The model to train or evaluate.
        """
        self.model = model
        self.device = None
        self._cache = {}
        self._iteration = None

    ##########################
    ### HIGH-LEVEL METHODS ###
    ##########################
    # pylint: disable=too-many-branches,too-many-statements
    def train(self, train_data, val_data=None, epochs=20, val_iterations=None,
              eval_every=None, eval_train=False, eval_val=True, callbacks=None,
              metrics=None, gpu='auto', **kwargs):
        """
        Method for training the model with the supplied parameters.

        Parameters
        ----------
        train_data: torch.DataLoader
            A data loader to obtain training data from. The samples yielded by the data loader
            depend on a specific trainer implementation.
        val_data: torch.DataLoader, default: None
            A data loader to use for validation. If the loader is an infinite data loader,
            `val_iterations` must also be given. If not supplied, no validation will be performed.
        epochs: int, default: 20
            The number of epochs to train for. If the given data is an infinite data loader, this
            value defines the number of iterations (and should most probably be increased).
        val_iterations: int, default: None
            The number of iterations to perform for validation. Must be given only if `val_data` is
            an infinite data loader. Otherwise, the parameter is ignored.
        eval_every: int, default: None
            How many iterations should pass until validation is called again. This should only be
            set if `train_data` is based on an iterable dataset which has infinite length. Note
            that everything works as if `epochs // eval_every` epochs is trained while eval_every
            can be considered the number of mini-batches.
        eval_train: bool, default: False
            Whether to compute metrics (apart from the loss) for both the validation and the train
            data. If this flag is set to False, metrics will only be computed for the validation
            data.
        eval_val: bool, default: True
            Whether to compute the loss (apart from the metrics) for the validation data. If this
            flag is set to False, no validation loss will be computed.
        callbacks: list of pyblaze.nn.TrainingCallback or
                pyblaze.nn.PredictionCallback, default: []
            The callbacks to use for training and inference. The training callbacks and prediction
            callbacks will be filtered automatically.
        metrics: dict of str -> func, default: {}
            Metrics to compute during evaluation for the validation data (and potentially for the
            training data). The keys for the metrics define their name.
        gpu: str or bool or int or list of int, default: 'auto'
            Governs, whether training and evaluation should be performed on a GPU. If set to True,
            the GPU with the most amount of free memory is selected (if there are multiple GPUs).
            If set to an integer, the GPU with the specified index is used. If set to a list of
            integers, the specified GPUs are used to train and evaluate the model und multiple GPUs
            simultaneously. In this case, the batch sizes of the data loaders should be adjusted
            accordingly. If set to a string, the only valid value is 'auto'. In this case, all
            available GPUs are used.
        kwargs: keyword arguments
            Additional keyword arguments dependent on the specific subclass.

        Returns
        -------
        pyblaze.nn.History
            A history object summarizing stats from the training. It contains as properties the
            development of the loss as `train_loss` (and potentially `val_loss`, if `val_data` is
            supplied and the keyword arguments include a function called `loss`). If additional
            metrics are supplied, there will be a property `val_<metric>` for each metric, and
            potentially `train_<metric>` if `eval_train` is set to True.
        """
        if metrics is None:
            metrics = {}
        if callbacks is None:
            callbacks = []

        # 1) Setup
        exception = None
        tic = time.time()

        if eval_every is not None:
            # Here, epochs are considered iterations
            epochs = epochs // eval_every

        # 1.1) Callbacks
        train_callbacks = [
            c for c in callbacks if isinstance(c, TrainingCallback)
        ]
        prediction_callbacks = [
            c for c in callbacks if isinstance(c, PredictionCallback)
        ]

        self._exec_callbacks(
            train_callbacks, 'before_training', self.model, epochs
        )

        # 1.2) Metrics
        metric_history = []
        if eval_val and 'loss' in kwargs:
            val_metrics = {**metrics, **{'loss': kwargs['loss']}}
        else:
            val_metrics = metrics

        # 1.3) Data loading
        if eval_every is not None:
            train_iterator = iter(train_data)

        # 1.4) GPU support
        gpu = self._gpu_descriptor(gpu)
        self._setup_device(gpu)
        self.model.to(self.device)

        # 2) Train for number of epochs
        for current_epoch in range(epochs):
            # 2.1) Prepare
            batch_iterations = eval_every or len(train_data)

            try:
                self._exec_callbacks(
                    train_callbacks, 'before_epoch', current_epoch,
                    batch_iterations
                )
            except CallbackException as e:
                exception = e
                break

            # 2.2) Train
            self.model.train()

            train_batch_weights = []
            train_losses = []

            if eval_every is not None:
                # 2.2.1) Iterable dataset
                for _ in range(eval_every):
                    item = next(train_iterator)
                    item = to_device(self.device, item)
                    loss = self.train_batch(item, **kwargs)
                    train_losses.append(loss)
                    self._exec_callbacks(
                        train_callbacks, 'after_batch', loss
                    )
            else:
                # 2.2.2) Dataset
                for item in train_data:
                    item = to_device(self.device, item)
                    loss = self.train_batch(item, **kwargs)
                    train_batch_weights.append(len(item))
                    train_losses.append(loss)
                    self._exec_callbacks(
                        train_callbacks, 'after_batch', loss
                    )

            # 2.3) Validate
            batch_metrics = Evaluation(self.collate_losses(train_losses), train_batch_weights)

            if val_data is not None:
                eval_val = self.evaluate(
                    val_data, iterations=val_iterations, metrics=val_metrics,
                    callbacks=prediction_callbacks, gpu=None
                ).with_prefix('val_')
                batch_metrics = Evaluation.merge(batch_metrics, eval_val)

            if eval_train:
                eval_train = self.evaluate(
                    train_data, iterations=val_iterations, metrics=metrics,
                    callbacks=prediction_callbacks, gpu=None
                ).with_prefix('train_')
                batch_metrics = Evaluation.merge(batch_metrics, eval_train)

            batch_metrics = Evaluation.merge(
                batch_metrics, Evaluation({'_timestamp': time.time()})
            )

            metric_history.append({
                'micro_train_loss': train_losses,
                **batch_metrics.to_dict()
            })

            # 2.4) Finish epoch
            try:
                self._exec_callbacks(
                    train_callbacks, 'after_epoch', batch_metrics
                )
            except CallbackException as e:
                exception = e
                break

        # 3) Finish training
        # 3.1) If GPU used
        if gpu is not None:
            self.model.to('cpu', non_blocking=True)
            self.device = None

        # 3.2) Finish callbacks
        self._exec_callbacks(
            train_callbacks, 'after_training'
        )
        if exception is not None:
            if isinstance(exception, CallbackException):
                exception.print()
            else:
                print(exception)

        return History(time.time() - tic, metric_history)

    def evaluate(self, data, iterations=None, metrics=None, callbacks=None, gpu='auto'):
        """
        Evaluates the model on the given data and computes the supplied metrics.

        Parameters
        ----------
        data: torch.DataLoader
            A data loader to obtain evaluation samples from. The expected samples depend on a
            specific trainer subclass.
        iterations: int, default: None
            The number of samples used for evaluating if the given data is an infinite data loader.
        metrics: dict of str -> func, default: {}
            The metrics to evaluate the model for. The keys define the names of the metrics when
            retrieving the evaluated result from the return parameter.
        callbacks: list of pyblaze.nn.PredictionCallback, default: []
            Callbacks to use while computing predictions. Usually, they are used for logging.
        gpu: str or bool or int or list of int, default: False
            Governs, whether training and evaluation should be performed on a GPU. If set to True,
            the GPU with the most amount of free memory is selected (if there are multiple GPUs).
            If set to an integer, the GPU with the specified index is used. If set to a list of
            integers, the specified GPUs are used to train and evaluate the model und multiple GPUs
            simultaneously. In this case, the batch sizes of the data loaders should be adjusted
            accordingly. If set to 'auto', all available GPUs are used. In case of `None`, the
            model will not be moved. Only use this option if you know what you are doing.

        Returns
        -------
        pyblaze.nn.training.wrappers.Evaluation
            An evaluation object, yielding as properties the metrics with their specified names.
        """
        if metrics is None:
            metrics = {}
        if callbacks is None:
            callbacks = []

        num_predictions = iterations or len(data)
        self._exec_callbacks(
            callbacks, 'before_predictions', self.model, num_predictions
        )

        if gpu is not None:
            gpu = self._gpu_descriptor(gpu)
            self._setup_device(gpu)
            self.model.to(self.device)

        self.model.eval()

        predictions = []
        targets = []

        iterator = iter(data)
        for _ in range(num_predictions):
            item = next(iterator)
            item = to_device(self.device, item)

            with torch.no_grad():
                prediction, target = self.eval_batch(item)

            predictions.append(to_device('cpu', prediction))
            targets.append(to_device('cpu', target))

            self._exec_callbacks(
                callbacks, 'after_batch', None
            )

        self._exec_callbacks(
            callbacks, 'after_predictions'
        )

        predictions = self.collate_predictions(predictions)
        targets = self.collate_targets(targets)

        def process_metric(k, m):
            if isinstance(m, dict):
                return [(f'{k}_{mk}', mm.item()) for mk, mm in m.items()]
            return [(k, m.item())]

        if targets is None:
            result = predictions
        else:
            result = dict(flatten(
                process_metric(k, f(predictions, targets))
                for k, f in metrics.items()
            ))

        if gpu is not None:
            self.model.to('cpu', non_blocking=True)
            self.device = None

        return Evaluation(result)

    def predict(self, data, iterations=None, callbacks=None, gpu='auto', **kwargs):
        """
        Computes predictions for the given samples.

        Parameters
        ----------
        data: torch.DataLoader
            The data loader from which to obtain items for prediction. Note that in most cases it
            is desirable that the data loader does *not* shuffle items as predictions may then be
            out-of-order.
        iterations: int, default: None
            The (maximum) number of samples used for evaluating if the given data is an iterable
            dataset.
        callbacks: list of pyblaze.nn.PredictionCallback
            Callbacks which are called as prediction progresses.
        gpu: str or bool or int or list of int, default: 'auto'
            Whether to use a (specific) GPU or multiple GPUs. If multiple GPUs are used, one
            process per GPU is started to minimize synchronization overhead. Make sure that using
            multiple GPUs makes up for this overhead. If `False` is specified, all cores of the
            computer are used to make predictions in parallel. In the case of 'auto', all available
            GPUs are used (if any).
        kwargs: keyword arguments
            Additional arguments passed directly to the `predict_batch` function.

        Returns
        -------
        object
            The predictions made by the model as collated by `collate_predictions`.
        """
        if callbacks is None:
            callbacks = []

        # 1) Set gpu if all is specified
        gpu = self._gpu_descriptor(gpu)

        # 2) Setup data loading
        num_iterations = iterations or len(data)

        self._exec_callbacks(
            callbacks, 'before_predictions', self.model, num_iterations
        )

        # 3) Make sure that the model is not data parallel, we don't need this for predicting
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model

        # 4) Now perform predictions
        if (isinstance(gpu, list) and len(gpu) > 1) or not gpu:
            # parallel computation
            if isinstance(gpu, list):
                num_workers = len(gpu)
            elif isinstance(gpu, bool):
                num_workers = os.cpu_count()
            else:
                num_workers = 1

            model.share_memory()

            callback = lambda: self._exec_callbacks(callbacks, 'after_batch', None)
            vectorizer = xmp.Vectorizer(
                _prediction_worker_func, _prediction_worker_init,
                callback_func=callback, num_workers=num_workers,
                gpu=gpu, model=model
            )

            iterator = iter(data)
            predictions = vectorizer.process(
                (next(iterator) for _ in range(num_iterations)),
                self.predict_batch, kwargs
            )
        else:
            # sequential computation
            device = gpu_device(gpu[0] if isinstance(gpu, list) else gpu)
            model = model.to(device)

            predictions = []

            iterator = iter(data)
            for _ in range(num_iterations):
                x = next(iterator)

                out = _prediction_worker_func(
                    x, self.predict_batch, kwargs, device
                )
                predictions.append(out)
                self._exec_callbacks(
                    callbacks, 'after_batch', None
                )

        self._exec_callbacks(
            callbacks, 'after_predictions'
        )

        return self.collate_predictions(predictions)

    ########################
    ### BATCH PROCESSING ###
    ########################
    @abstractmethod
    def train_batch(self, data, **kwargs):
        """
        Runs a single step in training. If the training data represents an infinite dataset, this
        equals a single iteration, otherwise a mini-batch.

        Parameters
        ----------
        data: object
            The data for the current iteration/mini-batch.

        Returns
        -------
        object
            The loss computed for the batch. If the returned value is not float, overwrite
            `collate_losses`.
        """

    @abstractmethod
    def eval_batch(self, data, **kwargs):
        """
        Runs a single step for inference. The data is either a mini-batch or a single iteration,
        depending on the data used for evaluation.

        Parameters
        ----------
        data: object
            The data for the current iteration/mini-batch.
        kwargs: keyword arguments
            Additional arguments dependent on the subclass and passed directly from the evaluation
            method.

        Returns
        -------
        torch.Tensor
            The output from the model.
        object
            The target, i.e. correct output. If this is not a `torch.Tensor`, the `collate_targets`
            should be overriden.
        """

    # pylint: disable=unused-argument
    def predict_batch(self, data, **kwargs):
        """
        Processes a single batch of inputs to make model predictions. The default implementation
        runs the given data through the model and does not perform any further actions. The outputs
        of this function will be collected for every batch and passed to the `collate_predictions`
        function.

        Note
        ----
        When using the `predict` function, this method is called within a `torch.no_grad()` block.

        Parameters
        ----------
        data: object
            The batch as sampled from the data loader.

        Returns
        -------
        object
            The return value passed to `collate_predictions`.
        """
        return self.forward(data)

    ###########################
    ### COLLATION FUNCTIONS ###
    ###########################
    def collate_predictions(self, predictions):
        """
        Combines the predictions obtained from the `eval_batch` function. The default
        implementation assumes that predictions are tensors and can simply be conatenated.

        Parameters
        ----------
        predictions: list of objects
            The predictions.

        Returns
        -------
        object
            An object to be used as predicted value for some metric.
        """
        return torch.cat(predictions)

    def collate_targets(self, targets):
        """
        Combines the targets in a way that it can be passed to some metric. The default
        implementation assumes that targets are tensors and simply concatenates them.

        Parameters
        ----------
        list of object
            The targets.

        Returns
        -------
        object
            An object to be used as true target with some metric.
        """
        return torch.cat(targets)

    def collate_losses(self, losses):
        """
        Combines the losses obtained from the `train_batch` function. The default implementation
        assumes that simple floats are returned. Overwriting this method might be useful if you
        have different types of losses, such as generator and discriminator loss when training GANs.

        Parameters
        ----------
        losses: list of object
            The losses returned from `train_batch`.

        Returns
        -------
        dict of str -> (float or list of float)
            The loss names mapped to their values.
        """
        return {'train_loss': losses}

    ######################
    ### BASE FUNCTIONS ###
    ######################
    def forward(self, x):
        """
        Computes the model output for common types of inputs.

        Parameters
        ----------
        x: list or tuple or dict or object
            The input the model.

        Returns
        -------
        object
            The output of the model (although arbitrary, it is usually a torch.Tensor).
        """
        if isinstance(x, (list, tuple)):
            return self.model(*x)
        if isinstance(x, dict):
            return self.model(**x)
        return self.model(x)

    def _gpu_descriptor(self, gpu):
        if gpu == 'auto':
            if cuda.device_count() == 0:
                return False
            if self.supports_multiple_gpus():
                return list(range(cuda.device_count()))
            return True
        return gpu

    def _setup_device(self, gpu):
        if isinstance(gpu, list) and len(gpu) > 1:
            self.model = xnnu.DataParallel(self.model, device_ids=gpu)
            self.device = gpu_device(gpu[0])
        else:
            self.device = gpu_device(gpu[0] if isinstance(gpu, list) else gpu)

    def _exec_callbacks(self, callbacks, func, *args):
        for callback in [self] + callbacks:
            getattr(callback, func)(*args)


def _prediction_worker_func(item, predict, kwargs, device):
    x = to_device(device, item)
    with torch.no_grad():
        out = predict(x, **kwargs)
    return to_device('cpu', out)


def _prediction_worker_init(rank, gpu, model):
    if isinstance(gpu, list):
        device = gpu_device(gpu[rank])
    else:
        device = gpu_device(gpu)
    model.to(device)
    return device
