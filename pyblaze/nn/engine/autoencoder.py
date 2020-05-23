from .mle import MLEEngine

class AutoencoderEngine(MLEEngine):
    """
    Utility class for easily initializing the :class:`pyblaze.nn.MLEEngine` for a usage with an
    autoencoder. This includes variational autoencoders.
    """

    def __init__(self, model, expects_data_target=False):
        """
        Initializes a new engine for autoencoders.

        Parameters
        ----------
        model: torch.nn.Module
            The model to train or evaluate.
        expects_data_target: bool, default: False
            Whether the data supplied to this engine is a tuple consisting of data and target or
            only contains the data.
        """
        super().__init__(
            model,
            expects_data_target=expects_data_target,
            uses_data_target=False,
            uses_data_input=True
        )
