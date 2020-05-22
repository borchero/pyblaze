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

Supervised Learning
-------------------

.. autoclass:: pyblaze.nn.LabelEngine
    :members:
    :exclude-members: __init__, after_batch, after_epoch, before_epoch, eval_batch, train_batch

Likelihood Learning
-------------------

.. autoclass:: pyblaze.nn.LikelihoodEngine
    :members:
    :exclude-members: eval_batch, train_batch

Variational Autoencoders
------------------------

.. autoclass:: pyblaze.nn.VAEEngine
    :members:
    :exclude-members: eval_batch, train_batch

Wasserstein GANs
----------------

.. autoclass:: pyblaze.nn.WGANEngine
    :members:
    :exclude-members: train_batch, eval_batch
