import copy
from abc import ABC
from collections import OrderedDict
import json
import torch
from pyblaze.utils.stdio import ensure_valid_directories

class Config(ABC):
    """
    Base class which standardizes configuration for PyTorch modules. For a specific configuration
    subclass, the configuration values are automatically deduced from the class's annotations. All
    annotations which do not define a default value are considered required.
    """

    @classmethod
    def load(cls, file):
        """
        Loads a configuration from the specified JSON file.

        Parameters
        ----------
        file: str
            The full path to the file. The extension does not need to be specified, but must be
            .json.

        Returns
        -------
        pyblaze.nn.Config
            The loaded configuration.
        """
        if not file.endswith('.json'):
            file += '.json'
        with open(file, 'r') as f:
            return cls(**json.load(f))

    @classmethod
    def _get_fields(cls):
        """
        Aggregates all available fields and their types from the class itself and its superclass
        (assuming it is a configuration as well).

        Returns
        -------
        dict
            Mapping from field names to types.
        """
        # pylint: disable=no-member
        annotations = {}
        if hasattr(super(), '_get_fields'):
            annotations = {**annotations, **super()._get_fields()}
        if hasattr(cls, '__annotations__'):
            annotations = {**annotations, **cls.__annotations__}
        return annotations

    def __init__(self, **kwargs):
        """
        Initializes a new config from the given parameters.
        """
        # 1) Get all available configuration options
        all_fields = type(self)._get_fields()
        keys = set(all_fields.keys())
        params = {}

        # 2) Set all parameters that are passed via the initializer
        for k, v in kwargs.items():
            if not k in keys:
                raise ValueError(
                    f'Unknown configuration parameter {k}.'
                )
            params[k] = v
            keys.remove(k)

        # 3) For all keys that are not set manually, assign default if available
        for k in keys:
            if not hasattr(type(self), k):
                raise ValueError(
                    f'Missing required configuration parameter {k}.'
                )
            params[k] = getattr(type(self), k)

        # 4) Check for all parameters if their types are correct
        for k, v in params.items():
            if not isinstance(v, all_fields[k]):
                raise ValueError(
                    f'Parameter {k} should have type {all_fields[k]} but '
                    f'has type {type(v)}.'
                )

        self._params = params

        # 5) Finally, set parameters on self and validate
        for k, v in params.items():
            setattr(self, k, v)

        if not self.is_valid():
            raise ValueError(
                f'Invalid configuration parameters. Check the documentation '
                f'for valid options.'
            )

    def save(self, file):
        """
        Saves the configuration to a JSON file.

        Parameters
        ----------
        file: str
            The full path to the file. The extension does not need to be specified, but must be
            .json.
        """
        ensure_valid_directories(file)
        if not file.endswith('.json'):
            file += '.json'
        with open(file, 'w+') as f:
            json.dump(self._params, f, indent=4, sort_keys=True)

    def is_valid(self):
        """
        Checks whether the given configuration is valid. This method should be overridden if
        required. The default implementation implies validity for all inputs matching the specified
        types.

        Returns
        -------
        bool
            Whether the configuration is valid.
        """
        return True

    def __repr__(self):
        return self._params.__repr__()


class Configurable(ABC):
    """
    Mixin class to make torch.nn.Module configurable in an easy and standardized way. For
    initialization, a module is then given a single config file of type `pyblaze.nn.Config`.
    Properties of the configuration can be easily accessed via `self.<property>`. Additionally,
    this class makes it easy to save and load the module in a consistent way. The binding to the
    configuration class is achieved by setting the __config__ class property to the class of the
    configuration.

    When subclassing a PyTorch module, make sure to include the Configurable mixin as *first*
    dependency.

    Note
    ----
    Only top-level modules should use this mixin to profit from saving/loading modules easily.
    """

    @classmethod
    def load(cls, file):
        """
        Loads the model by loading a configuration defining the architecture and another file
        containing the weights for this architecture.

        Parameters
        ----------
        file: str
            The full path to the file. The configuration will be loaded from <file>.json and the
            model's weights will be loaded from <file>.pt.

        Returns
        -------
        pyblaze.nn.Configurable
            The loaded model.
        """
        if hasattr(cls, '__config__'):
            config = cls.__config__.load(file)
            model = cls(config)
        else:
            model = cls(None)
        params = torch.load(f'{file}.pt')
        model.load_state_dict(params)
        return model

    def __init__(self, *args, **kwargs):
        """
        Initializes the configurable model. You can either pass an instance of the model's
        configuration class or the keyword arguments which you would normally pass to initialize
        the configuration instance. The configuration is then created implicitly.

        Parameters
        ----------
        args: variadic arguments
            Either zero or one parameters. If one is given, it must be an instance of the model's
            configuration class.
        kwargs: keyword arguments
            The arguments passed to the initializer of the configuration class. Must only be given
            if the configuration instance is not defined.
        """
        super(Configurable, self).__init__()

        assert len(args) <= 1, \
            "The only variadic argument must be a configuration instance."

        if len(args) == 1:
            assert len(kwargs) == 0, \
                "Variadic arguments must not be given if a another " + \
                "argument is defined."
            assert args[0] is None or \
                isinstance(args[0], type(self).__config__), \
                "Given configuration does not match the model's config class."

            self._config = args[0]
        else:
            self._config = type(self).__config__(**kwargs)

    @property
    def num_parameters(self):
        """
        Returns the number of parameters that this model contains.

        Returns
        -------
        int
            The number of parameters.
        """
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self):
        """
        Returns the number of trainable parameters that this model contains.

        Returns
        -------
        int
            The number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, file):
        """
        Saves the model and its configuration to two files.

        Parameters
        ----------
        file: str
            The full path to the file. The configuration will be saved to <file>.json while the
            model's weights will be saved to <file>.pt.
        """
        # pylint: disable=protected-access
        ensure_valid_directories(file)
        self._config.save(file)
        # Ensure saving CPU model
        state_dict = copy.deepcopy(self.state_dict())
        result = OrderedDict()
        for k, v in state_dict.items():
            result[k] = v.cpu()
        result._metadata = state_dict._metadata
        torch.save(result, f'{file}.pt')

    def __getattr__(self, name):
        try:
            return super(Configurable, self).__getattr__(name)
        except AttributeError:
            if name == '_config':
                raise AttributeError()
            return getattr(self._config, name)
