Callbacks
=========

.. code:: python

    import pyblaze.nn as xnn

The callback module exposes a variety of callbacks that may be used in conjunction with some
:class:`Engine`. The base classes further enable the definition of custom callbacks.

.. contents:: Contents
    :local:
    :depth: 1

Base Classes
------------

Exception
^^^^^^^^^

.. autoclass:: pyblaze.nn.CallbackException
    :members:

Training Callback
^^^^^^^^^^^^^^^^^

.. autoclass:: pyblaze.nn.TrainingCallback
    :members:

Value Training Callback
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyblaze.nn.ValueTrainingCallback
    :members:

Prediction Callback
^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyblaze.nn.PredictionCallback
    :members:

Logging
-------

Epoch Progress
^^^^^^^^^^^^^^

.. autoclass:: pyblaze.nn.EpochProgressLogger
    :members:
    :exclude-members: before_training, after_epoch, after_training

Batch Progress
^^^^^^^^^^^^^^

.. autoclass:: pyblaze.nn.BatchProgressLogger
    :members:
    :exclude-members: before_training, before_epoch, after_batch, after_epoch, after_training

Prediction Progress
^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyblaze.nn.PredictionProgressLogger
    :members:
    :exclude-members: before_prediction, after_batch, after_predictions

Checkpointing
-------------

Model Saving
^^^^^^^^^^^^

.. autoclass:: pyblaze.nn.ModelSaverCallback
    :members:
    :exclude-members: after_epoch, after_training, before_epoch, before_training

Scheduling
----------

Learning Rate
^^^^^^^^^^^^^

.. autoclass:: pyblaze.nn.LearningRateScheduler
    :members:
    :exclude-members: after_batch, after_epoch

Parameter
^^^^^^^^^

.. autoclass:: pyblaze.nn.ParameterScheduler
    :members:
    :exclude-members: read, after_batch, after_epoch, after_training, before_epoch, before_training

Early Stopping
^^^^^^^^^^^^^^

.. autoclass:: pyblaze.nn.EarlyStopping
    :members:
    :exclude-members: after_epoch, after_training, before_training

Tracking
--------

Neptune
^^^^^^^

.. autoclass:: pyblaze.nn.NeptuneTracker
    :members:
    :exclude-members: after_batch, after_epoch

Tensorboard
^^^^^^^^^^^

.. autoclass:: pyblaze.nn.TensorboardTracker
    :members:
    :exclude-members: before_training, after_batch, after_epoch
