from .mle import MLEEngine
from ._utils import forward

class AutoencoderEngine(MLEEngine):
    """
    Utility class for easily initializing the :class:`pyblaze.nn.MLEEngine` for a usage with an
    autoencoder. This includes variational autoencoders.

    When making use of the :meth:`predict` method, the model passed to this engine should have
    a submodule named :code:`decoder`. The :meth:`predict` method takes the following additional
    parameters:

    Parameters
    ----------
    reconstruct: bool, default: False
        If this flag is set, data passed to the :meth:`predict` method is assumed to be the input to
        the encoder. Otherwise, the input is assumed to be from the latent space, i.e. to be the
        direct input to the decoder. In both cases, the output of the decoder is returned to the
        caller.
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

    def predict_batch(self, data, reconstruct=False):
        if reconstruct:
            x, _ = self._get_x_target(data)
            return forward(self.model, x)
        return forward(self.model.decoder, data)
