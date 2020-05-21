Neural Network Training
=======================

Engine
------

Base
^^^^

.. automodule:: pyblaze.nn.engine.base
    :members:

Supervised Learning
^^^^^^^^^^^^^^^^^^^

.. automodule:: pyblaze.nn.engine.label
    :members:
    :exclude-members: __init__, after_batch, after_epoch, before_epoch, eval_batch, train_batch

Variational Autoencoders
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pyblaze.nn.engine.vae
    :members:
    :exclude-members: eval_batch, train_batch

Wasserstein GANs
^^^^^^^^^^^^^^^^

.. automodule:: pyblaze.nn.engine.wgan
    :members:
    :exclude-members: train_batch, eval_batch

Callbacks
---------

Base
^^^^

.. automodule:: pyblaze.nn.callbacks.base
    :members:

Logging
^^^^^^^

.. automodule:: pyblaze.nn.callbacks.logging
    :members:
    :exclude-members: after_batch, after_epoch, after_predictions, after_training, before_epoch, before_predictions, before_training

Saving
^^^^^^

.. automodule:: pyblaze.nn.callbacks.saving
    :members:
    :exclude-members: after_epoch, after_training, before_epoch, before_training
    

Scheduling
^^^^^^^^^^

.. automodule:: pyblaze.nn.callbacks.schedule
    :members:
    :exclude-members: after_batch, after_epoch, after_training, before_epoch, before_training

Early Stopping
^^^^^^^^^^^^^^

.. automodule:: pyblaze.nn.callbacks.early_stopping
    :members:
    :exclude-members: after_epoch, after_training, before_training

Tracking
^^^^^^^^

.. automodule:: pyblaze.nn.callbacks.tracking
    :members:
    :exclude-members: after_batch, after_epoch

Modules
-------

Neural Network Layers
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pyblaze.nn.modules.lstm
    :members:

Losses
^^^^^^

.. automodule:: pyblaze.nn.modules.vae
    :members:

.. automodule:: pyblaze.nn.modules.wasserstein
    :members:

.. automodule:: pyblaze.nn.modules.gp
    :members:

Functional
----------

Metrics
^^^^^^^

.. automodule:: pyblaze.nn.functional.metrics
    :members:

Others
^^^^^^

.. automodule:: pyblaze.nn.functional.gumbel
    :members:

.. automodule:: pyblaze.nn.functional.random
    :members:

Utils
-----

Estimator
^^^^^^^^^
.. automodule:: pyblaze.nn.utils.estimator
    :members:

Config
^^^^^^

.. automodule:: pyblaze.nn.utils.config
    :members:

Wrappers
^^^^^^^^

.. automodule:: pyblaze.nn.engine._history
    :members:
    :exclude-members: __init__, before_training, after_training, after_batch, after_epoch
