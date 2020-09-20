Data
====

.. code:: python

    import pyblaze.nn as xnn

The data module provides utilities for working with PyTorch datasets.

.. contents:: Contents
    :local:
    :depth: 1

Extensions
----------

Deterministic Splitting
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyblaze.nn.data.extensions.split

Random Splitting
^^^^^^^^^^^^^^^^

.. autofunction:: pyblaze.nn.data.extensions.random_split

Data Loader
^^^^^^^^^^^

.. autofunction:: pyblaze.nn.data.extensions.loader

Datasets
--------

Transform
^^^^^^^^^

.. autoclass:: pyblaze.nn.NoiseDataset
    :members:
    :inherited-members:
    :exclude-members: loader, split, random_split

Noise
^^^^^

.. autoclass:: pyblaze.nn.NoiseDataset
    :members:
    :inherited-members:
    :exclude-members: loader, split, random_split

.. autoclass:: pyblaze.nn.LabeledNoiseDataset
    :members:
    :inherited-members:
    :exclude-members: loader, split, random_split

Data Loaders
------------

Zipping
^^^^^^^

.. autoclass:: pyblaze.nn.ZipDataLoader
    :members:
    :inherited-members:
