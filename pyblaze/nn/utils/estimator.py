from pyblaze.utils.stdlib import cached_property

class Estimator:
    """
    Estimators are meant to be mixins for PyTorch modules. They extend the module with three
    additional methods:

    - `fit(...)`
        This function optimizes the parameters of the model given some (input and output) data.
    - `evaluate(...)`
        This function estimates the performance of the model by returning some suitable metric
        based on some (input and output) data. This metric is usually used in the `fit` method as
        well (e.g. an appropriate loss).
    - `predict(...)`
        This function performs inference based on some (input) data. This method is usually tightly
        coupled with the model's forward method, however, opens additional possibilities such as
        easy GPU support.

    Usually, the module does not implement these method itself, *unless training is tightly coupled
    with the model parameters*. An example might be a linear regression module. Normally, however,
    the module is expected to provide an `__engine__` class that accepts the module as first
    parameter (or alternatively overwrite the `engine` cached property). This engine class acts as
    a default engine which is used whenever one of the methods is called. The arguments for the
    functions therefore depend on the particular engine that is being used.
    """

    @cached_property
    def engine(self):
        """
        Returns the engine for this model. The engine is cached after the first access to this
        property. The class of the engine has to be given by `__engine__`.

        Returns
        -------
        pyblaze.nn.BaseEngine
            The `__engine__` class initialized with this model.
        """
        # pylint: disable=no-member
        return type(self).__engine__(self)

    def fit(self, *args, **kwargs):
        """
        Optimizes the parameters of the model based on input and output data. The parameters are
        passed to the `train` method of the model's engine.
        """
        assert len(args) > 0 or 'data' in kwargs, \
            "Data must be given as keyword argument or first parameter."

        if len(args) > 0:
            data = self.prepare_input(args[0])
            args = args[1:]
        else:
            data = self.prepare_input(kwargs['data'])
            del kwargs['data']

        return self.engine.train(data, *args, **kwargs)

    def evaluate(self, *args, **kwargs):
        """
        Estimates the performance of the model by returning some metric based on input and output
        data. The parameters are passed to the `evaluate` method of the model's engine.
        """
        assert len(args) > 0 or 'data' in kwargs, \
            "Data must be given as keyword argument or first parameter."

        if len(args) > 0:
            data = self.prepare_input(args[0])
            args = args[1:]
        else:
            data = self.prepare_input(kwargs['data'])
            del kwargs['data']

        return self.engine.evaluate(data, *args, **kwargs)

    def predict(self, *args, **kwargs):
        """
        Performs inference based on some input data. The parameters are passed to the `predict`
        method of the model's engine.
        """
        assert len(args) > 0 or 'data' in kwargs, \
            "Data must be given as keyword argument or first parameter."

        if len(args) > 0:
            data = self.prepare_input(args[0])
            args = args[1:]
        else:
            data = self.prepare_input(kwargs['data'])
            del kwargs['data']

        return self.engine.predict(data, *args, **kwargs)

    def prepare_input(self, data):
        """
        Prepares the input for the engine. This enables passing other types of data instead of
        PyTorch data loaders when it is appropriate, making it easier to e.g. provide a Sklearn-
        like interface. By default, the data object is simply returned but subclasses may override
        this function as appropriate.

        Parameters
        ----------
        data: object
            The data object passed to `fit`, `evaluate` or `predict`.

        Returns
        -------
        iterable
            The iterable dataset to use for the engine's data.
        """
        return data
