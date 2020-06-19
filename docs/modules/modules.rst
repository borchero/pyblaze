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

Normalizing Flows
-----------------

Normalizing Flow
^^^^^^^^^^^^^^^^

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
