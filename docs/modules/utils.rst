Utils
=====

.. code:: python

    import pyblaze.nn as xnn
    import pyblaze.utils as U

This utility documentation encompasses contents from two modules, namely the :code:`nn.utils` as
well as the :code:`utils` module.

.. contents:: Contents
    :local:
    :depth: 1

Neural Networks
---------------

Estimator
^^^^^^^^^

.. autoclass:: pyblaze.nn.Estimator
    :members:

Config
^^^^^^

.. autoclass:: pyblaze.nn.Config
    :members:

Training History
----------------

.. autoclass:: pyblaze.nn.engine._history.History
    :members:
    :exclude-members: __init__, before_training, after_training, after_batch, after_epoch

Progress
--------

.. autoclass:: pyblaze.utils.ProgressBar
    :members:
