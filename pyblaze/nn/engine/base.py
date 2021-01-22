from abc import ABC, abstractmethod
from collections import defaultdict
import torch
import torch.nn as nn
import torch.cuda as cuda
from pyblaze.nn.callbacks import TrainingCallback, PredictionCallback, CallbackException, \
    ValueTrainingCallback
import pyblaze.nn.utils as xnnu
from pyblaze.utils.torch import gpu_device, to_device
from ._history import History
from ._utils import forward

class Engine(TrainingCallback, PredictionCallback, ABC):
    """
    A base class for training and evaluating models as well as making predictions. Generally, this
    class should be seen as _binding_ between a model and some data. Models should always be
    wrapped in an engine, both when being trained, and when performing inference/evaluation. The
    engine ensures that the model's environment is correct and prevents plenty of pitfalls.

    A concrete implementation of this class is tailored to a specific type of data (e.g.
    independent, identically distributed data samples) and/or model types (e.g. GANs).
    """

    #----------------------------------------------------------------------------------------------
    # ENGINE CONFIGURATION
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

    #----------------------------------------------------------------------------------------------
    # INITIALIZATION
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

    #----------------------------------------------------------------------------------------------
    # TRAINING
    # pylint: disable=too-many-branches,too-many-statements
    def train(self, train_data, val_data=None, epochs=20, val_iterations=None, eval_every=None,
              eval_train=False, eval_val=True, callbacks=None, metrics=None, gpu='auto', **kwargs):
        r"""
        Method for training the model with the supplied parameters.

        Parameters
        ----------
        train_data: torch.utils.data.DataLoader
            A data loader to obtain training data from. The samples yielded by the data loader
            depend on a specific engine implementation.
        val_data: torch.utisl.data..DataLoader, default: None
            A data loader to use for validation. If the loader is an infinite data loader,
            `val_iterations` must also be given. If not supplied, no validation will be performed.
        epochs: int, default: 20
            The number of epochs to train for. If the given data is an infinite data loader, this
            value defines the number of iterations (and should most probably be increased).
        val_iterations: int, default: None
            The number of iterations to perform for validation. Must be given only if `val_data` is
            an infinite data loader. Otherwise, the parameter is ignored.
        eval_every: int, default: None
            How many iterations should pass until validation is called again. If this is set when
            `train_data` is an iterable dataset, then it defines the number of iterations per
            epoch (i.e. before performing validation again). Otherwise, it defines the number of
            epochs to train before calling validation.
        eval_train: bool, default: False
            Whether to compute metrics (apart from the loss) for both the validation and the train
            data. If this flag is set to False, metrics will only be computed for the validation
            data.
        eval_val: bool, default: True
            Whether to compute the loss (apart from the metrics) for the validation data. If this
            flag is set to `False`, no validation loss will be computed.
        callbacks: list of (pyblaze.nn.TrainingCallback or pyblaze.nn.PredictionCallback),
                default: []
            The callbacks to use for training and inference. The training callbacks and prediction
            callbacks will be filtered automatically.
        metrics: dict of str -> func, default: {}
            Metrics to compute during evaluation for the validation data (and potentially for the
            training data). The keys for the metrics define their name.
        gpu: str or bool or int or list of int, default: 'auto'
            Governs, whether training and evaluation should be performed on a GPU. If set to True,
            the first GPU is selected. If set to an integer, the GPU with the specified index is
            used. If set to a list of integers, the specified GPUs are used to train and evaluate
            the model und multiple GPUs simultaneously. In this case, the batch sizes of the data
            loaders should be adjusted accordingly. If set to a string, the only valid value is
            'auto'. In this case, all available GPUs are used.
        kwargs: keyword arguments
            Additional keyword arguments dependent on the specific subclass. If prefixed with
            'eval\_', it will be passed to :meth:`eval_batch` (without the prefix), otherwise to
            :meth:`train_batch`. All keyword arguments may also be given as `ParameterScheduler`
            to enable dynamic changes in parameters. Note that the scheduler can also be used to
            alter values for 'eval\_' keywords. However, they do so following the same schedule as
            the training keywords. In any case, consult the engine's documentation if the
            parameter allows dynamic behavior. Semantically, it might not make sense to do.

        Returns
        -------
        pyblaze.nn.History
            A history object summarizing stats from the training.
        """
        if metrics is None:
            metrics = {}
        if callbacks is None:
            callbacks = []

        # 1) Setup
        try:
            batch_iterations = len(train_data)
            iterable_data = False
        except: # pylint: disable=bare-except
            batch_iterations = eval_every
            iterable_data = True

        exception = None
        if iterable_data and eval_every is not None:
            # Here, epochs are considered iterations
            epochs = epochs // eval_every

        # 1.1) Callbacks
        history = History()
        # Prepend the engine's callbacks to the passed callbacks
        callbacks = [history] + callbacks
        # Also, add the callbacks that are extracted from the keyword arguments
        callbacks += [v for _, v in kwargs.items() if isinstance(v, TrainingCallback)]
        # Then, we can extract the callbacks for training and prediction
        train_callbacks = [c for c in callbacks if isinstance(c, TrainingCallback)]
        prediction_callbacks = [c for c in callbacks if isinstance(c, PredictionCallback)]
        self._exec_callbacks(train_callbacks, 'before_training', self.model, epochs)

        # 1.2) Metrics
        if eval_val and 'loss' in kwargs:
            val_metrics = {**metrics, **{'loss': kwargs['loss']}}
        else:
            val_metrics = metrics

        # 1.3) Data loading
        if iterable_data:
            train_iterator = iter(train_data)

        # 1.4) GPU support
        gpu = self._gpu_descriptor(gpu)
        self._setup_device(gpu)
        self.model.to(self.device)

        # 1.5) Valid kwargs
        train_kwargs = {k: v for k, v in kwargs.items() if not k.startswith('eval_')}
        dynamic_train_kwargs = {
            k: v for k, v in train_kwargs.items() if isinstance(v, ValueTrainingCallback)
        }
        eval_kwargs = {k[5:]: v for k, v in kwargs.items() if k.startswith('eval_')}
        dynamic_eval_kwargs = {
            k: v for k, v in eval_kwargs.items() if isinstance(v, ValueTrainingCallback)
        }

        # 2) Train for number of epochs
        for current_epoch in range(epochs):
            # 2.1) Prepare
            try:
                self._exec_callbacks(
                    train_callbacks, 'before_epoch', current_epoch, batch_iterations
                )
            except CallbackException as e:
                exception = e
                break

            # 2.2) Train
            self.model.train()

            batch_losses = []
            if not iterable_data:
                train_iterator = iter(train_data)

            for _ in range(batch_iterations):
                train_kwargs = {
                    **train_kwargs,
                    **{k: v.read() for k, v in dynamic_train_kwargs.items()}
                }
                item = next(train_iterator)
                item = self.to_device(self.device, item)
                loss = self.train_batch(item, **train_kwargs)
                batch_losses.append(loss)
                self._exec_callbacks(train_callbacks, 'after_batch', _strip_metrics(loss))

            # 2.3) Validate
            epoch_metrics = self.collate_losses(batch_losses)
            eval_kwargs = {
                **eval_kwargs,
                **{k: v.read() for k, v in dynamic_eval_kwargs.items()}
            }
            do_val = eval_every is None or iterable_data or \
                current_epoch % eval_every == 0 or current_epoch == epochs - 1

            if val_data is not None and do_val:
                eval_metrics = self.evaluate(
                    val_data, iterations=val_iterations, metrics=val_metrics,
                    callbacks=prediction_callbacks, gpu=None, **eval_kwargs
                )
                epoch_metrics = {
                    **epoch_metrics, **{f'val_{k}': v for k, v in eval_metrics.items()}
                }

            if eval_train and do_val:
                eval_metrics = self.evaluate(
                    train_data, iterations=val_iterations, metrics=metrics,
                    callbacks=prediction_callbacks, gpu=None, **eval_kwargs
                )
                epoch_metrics = {
                    **epoch_metrics, **{f'train_{k}': v for k, v in eval_metrics.items()}
                }

            # 2.4) Finish epoch
            try:
                self._exec_callbacks(train_callbacks, 'after_epoch', epoch_metrics)
            except CallbackException as e:
                exception = e
                break

        # 3) Finish training
        # 3.1) If GPU used
        if gpu is not None:
            self.model.to('cpu', non_blocking=True)
            self.device = None

        # 3.2) Finish callbacks
        self._exec_callbacks(train_callbacks, 'after_training')
        if exception is not None:
            if isinstance(exception, CallbackException):
                exception.print()
            else:
                print(exception)

        return history

    #----------------------------------------------------------------------------------------------
    # EVALUATION
    def evaluate(self, data, iterations=None, metrics=None, callbacks=None, gpu='auto', **kwargs):
        """
        Evaluates the model on the given data and computes the supplied metrics.

        Parameters
        ----------
        data: torch.DataLoader
            A data loader to obtain evaluation samples from. The expected samples depend on a
            specific engine subclass.
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
        kwargs: keyword arguments
            Additional arguments, passed directly to the `eval_batch` method.

        Returns
        -------
        pyblaze.nn.training.wrappers.Evaluation
            An evaluation object, yielding as properties the metrics with their specified names.
        """
        if metrics is None:
            metrics = {}
        if callbacks is None:
            callbacks = []

        # Setup
        num_predictions = iterations or len(data)
        self._exec_callbacks(callbacks, 'before_predictions', self.model, num_predictions)

        # Ensure GPU
        if gpu is not None:
            gpu = self._gpu_descriptor(gpu)
            self._setup_device(gpu)
            self.model.to(self.device)

        # Run inference
        self.model.eval()

        evals = []
        iterator = iter(data)
        for _ in range(num_predictions):
            item = next(iterator)
            item = self.to_device(self.device, item)

            with torch.no_grad():
                eval_out = self.eval_batch(item, **kwargs)

            evals.append(self.to_device('cpu', eval_out))
            self._exec_callbacks(callbacks, 'after_batch', None)

        self._exec_callbacks(callbacks, 'after_predictions')

        evals = self.collate_evals(evals)

        if gpu is not None:
            self.model.to('cpu', non_blocking=True)
            self.device = None

        return {
            key: self._process_metric(forward(metric, evals))
            for key, metric in metrics.items()
        }

    #----------------------------------------------------------------------------------------------
    # PREDICTIONS
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

        self._exec_callbacks(callbacks, 'before_predictions', self.model, num_iterations)

        # 3) Make sure that the model is not data parallel, we don't need this for predicting
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model

        # 4) Now perform predictions sequentially
        device = gpu_device(gpu[0] if isinstance(gpu, list) else gpu)
        model = model.to(device)

        predictions = []
        iterator = iter(data)
        for _ in range(num_iterations):
            x = next(iterator)
            x = self.to_device(device, x)
            with torch.no_grad():
                out = self.predict_batch(x, **kwargs)
            out = self.to_device('cpu', out)
            predictions.append(out)
            self._exec_callbacks(callbacks, 'after_batch', None)

        self._exec_callbacks(callbacks, 'after_predictions')

        return self.collate_predictions(predictions)

    #----------------------------------------------------------------------------------------------
    # BATCH PROCESSING
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
            :meth:`collate_losses`. If you return a dict here, all keys prefixed with an underscore
            will not be passed to any callbacks but only be available in the :meth:`collate_losses`
            method.
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
        object
            The model output and possibly some correct target. If this is not a
            :code:`torch.Tensor`, the :meth:`collate_evals`.
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
        return forward(self.model, data)

    #----------------------------------------------------------------------------------------------
    # COLLATION FUNCTIONS
    def collate_losses(self, losses):
        """
        Combines the losses obtained from the :meth:`train_batch` method. The default implementation
        assumes that simple floats are returned. Overwriting this method might be useful if you
        have different types of losses, such as generator and critic loss when training Wasserstein
        GANs.

        Note
        ----
        Overwrite this function if it is important to weigh losses obtained from batches in some
        manner (e.g. if data batches have different size).

        Parameters
        ----------
        losses: list of object
            The losses returned from `train_batch`.

        Returns
        -------
        dict of str -> object
            The loss names mapped to their values (usually float values).
        """
        ref = losses[0]

        if isinstance(ref, dict):
            result = defaultdict(list)
            for item in losses:
                for k, v in item.items():
                    result[k].append(v)
            return {k: sum(v) / len(v) for k, v in result.items()}

        if isinstance(ref, (list, tuple)):
            result = [[] for _ in len(ref)]
            for item in losses:
                for i, it in enumerate(item):
                    result[i].append(it)
            return {f'loss_{i}': sum(r) / len(r) for i, r in enumerate(result)}

        return {'loss': sum(losses) / len(losses)}

    def collate_evals(self, evals):
        """
        Combines the evaluation objects returned from the :meth:`eval_batch` method. The default
        implementation works whenever the returned objects are tensors, tuples of tensors or
        dictionaries of tensors. The contents will then simply be concatenated.

        Parameters
        ----------
        evals: list of objects
            The evaluation objects.

        Returns
        -------
        object
            An object that is passed to all metrics. If a tuple or a dict, they are passed as
            variadic arguments or keyword arguments, respectively.
        """
        return self._collate(evals)

    def collate_predictions(self, predictions):
        """
        Combines the predictions obtained for multiple batches as per the :meth:`predict_batch`
        method. The default implementation works whenever the returned objects are tensors, tuples
        of tensors or dictionaries of tensors. The contents will then simply be concatenated.

        Parameters
        ----------
        predictions: list of objects
            The predictions obtained.

        Returns
        -------
        object
            An object that is obtained from the predictions.
        """
        return self._collate(predictions)

    #----------------------------------------------------------------------------------------------
    # OVERRIDABLE UTILITY FUNCTIONS
    def to_device(self, device, item):
        """
        This method moves the given item to the provided device. The default implementation allows
        for Tensors as well as dicts, lists and tuples of tensors (they may be nested). If you
        require passing custom datatypes to a device as you feed them to your model or your model
        outputs them, you may overwrite this method.

        Parameters
        ----------
        device: torch.Device or str
            The device onto which the given item should be moved. May either be an actual device
            object (such as a GPU) or a string identifying the device (e.g. 'cpu' or 'cuda:0').
        item: object
            The item to move.

        Returns
        -------
        object
            The item moved to the given device.
        """
        return to_device(device, item)

    #----------------------------------------------------------------------------------------------
    # UTILITY FUNCTIONS
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

    def _collate(self, items):
        ref = items[0]
        if isinstance(ref, dict):
            return {key: torch.cat([v[key] for v in items]) for key in ref.keys()}
        # recursive call of _collate for a more generic functionality
        if isinstance(ref, (list, tuple)):
            return tuple(self._collate([v[i] for v in items]) for i in range(len(ref)))
        return torch.cat(items)

    def _process_metric(self, metric):
        if isinstance(metric, torch.Tensor):
            if metric.numel() == 1:
                return metric.item()
            return metric
        return metric


def _strip_metrics(metrics):
    # If `metrics` is a dict, we remove all keys starting with an underscore
    if isinstance(metrics, dict):
        return {k: v for k, v in metrics.items() if not k.startswith('_')}
    return metrics


def _prediction_worker_init(rank, gpu, model):
    if isinstance(gpu, list):
        device = gpu_device(gpu[rank])
    else:
        device = gpu_device(gpu)
    model.to(device)
    return device
