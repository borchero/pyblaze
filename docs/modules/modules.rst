Modules
=======

.. code:: python

    import pyblaze.nn as xnn

The modules module provides a variety of neural network layers that are not included directly in
PyTorch. They are simply implementations of :code:`torch.nn.Module`.

.. contents:: Contents
    :local:
    :depth: 1

Basic
-----

Stacked LSTM
^^^^^^^^^^^^

.. autoclass:: pyblaze.nn.StackedLSTM
    :members:

Stacked LSTM Cell
^^^^^^^^^^^^^^^^^

.. autoclass:: pyblaze.nn.StackedLSTMCell
    :members:

Linear Residual
^^^^^^^^^^^^^^^

.. autoclass:: pyblaze.nn.LinearResidual
    :members:

View
^^^^

.. autoclass:: pyblaze.nn.View
    :members:

Variational Autoencoder
-----------------------

Loss
^^^^

.. autoclass:: pyblaze.nn.VAELoss
    :members:

Wasserstein GANs
----------------

Generator Loss
^^^^^^^^^^^^^^

.. autoclass:: pyblaze.nn.WassersteinLossGenerator
    :members:

Critic Loss
^^^^^^^^^^^

.. autoclass:: pyblaze.nn.WassersteinLossCritic
    :members:

Gradient Penalty
^^^^^^^^^^^^^^^^

.. autoclass:: pyblaze.nn.GradientPenalty
    :members:

Density Estimation
------------------

Masked Autoencoder
^^^^^^^^^^^^^^^^^^

.. autoclass:: pyblaze.nn.MADE
    :members:

Normal Loss
^^^^^^^^^^^

.. autoclass:: pyblaze.nn.TransformedNormalLoss
    :members:

GMM Loss
^^^^^^^^

.. autoclass:: pyblaze.nn.TransformedGmmLoss
    :members:

Normalizing Flows
-----------------

.. autoclass:: pyblaze.nn.NormalizingFlow
    :members:

Affine Transform
^^^^^^^^^^^^^^^^

.. autoclass:: pyblaze.nn.AffineTransform
    :members:

Planar Transform
^^^^^^^^^^^^^^^^

.. autoclass:: pyblaze.nn.PlanarTransform
    :members:

Radial Transform
^^^^^^^^^^^^^^^^

.. autoclass:: pyblaze.nn.RadialTransform
    :members:

Affine Coupling Transform 1D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyblaze.nn.AffineCouplingTransform1d
    :members:

Masked Autoregressive Transform 1D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyblaze.nn.MaskedAutoregressiveTransform1d
    :members:

BatchNorm Transform 1D
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyblaze.nn.BatchNormTransform1d
    :members:
