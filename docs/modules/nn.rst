Neural Network Training
=======================

Data
----

.. automodule:: pyblaze.nn.data.noise
    :members:
    :inherited-members:

Engine
------

Base
^^^^

.. automodule:: pyblaze.nn.engine.base
    :members:

Implementation
^^^^^^^^^^^^^^

.. automodule:: pyblaze.nn.engine.supervised
    :members:
    :exclude-members: __init__, after_batch, after_epoch, before_epoch, eval_batch, train_batch

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

.. automodule:: pyblaze.nn.modules.lstm
    :members:

.. automodule:: pyblaze.nn.modules.loss
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

.. automodule:: pyblaze.nn.engine.wrappers
    :members:
    :exclude-members: __init__, cat, with_prefix, merge
