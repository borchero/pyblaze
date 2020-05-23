Engines
=======

.. code:: python

    import pyblaze.nn as xnn

The engine module is PyBlaze's most central component as it provides the engines driving the
training and evaluation loops. The base class enables implementing custom engines. However, the
available engines are already usable for a variety of use cases.

.. contents:: contents
    :local:
    :depth: 1

Base Class
----------

.. autoclass:: pyblaze.nn.Engine
    :members:

Maximum Likelihood Estimation
-----------------------------

.. autoclass:: pyblaze.nn.MLEEngine
    :members:
    :exclude-members: after_batch, after_epoch, before_epoch, eval_batch, train_batch

Autoencoders
------------

.. autoclass:: pyblaze.nn.AutoencoderEngine
    :members:

Wasserstein GANs
----------------

.. autoclass:: pyblaze.nn.WGANEngine
    :members:
    :exclude-members: train_batch, eval_batch
