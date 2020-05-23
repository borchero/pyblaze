PyBlaze Documentation
=====================

.. image:: https://img.shields.io/pypi/v/pyblaze?label=version
.. image:: https://img.shields.io/github/license/borchero/pyblaze?label=license

`PyBlaze <https://github.com/borchero/pyblaze>`_ is a high-level library for large-scale machine learning in `PyTorch <https://pytorch.org>`_. It is engineered to cut obsolete boilerplate code while preserving the flexibility of PyTorch to create just about any deep learning model.

Installation
------------

PyBlaze is available on PyPi and can simply be installed as follows:

.. code:: bash

    pip install pyblaze

Library Design
--------------

PyBlaze revolves around the concept of an **engine**. An engine is a powerful abstraction for
combining a model's definition with the algorithm required to optimize its parameters according to
some data. Engines provided by PyBlaze are focused on generalization: while the engine encapsulates
the optimization algorithm, the user must explicitly define the optimization objective (usually the
loss function).

However, engines go far beyond implementing the optimization algorithm. Specifically, they further
provide the following features:

- **Evaluation**: During training, validation data can be used to evaluate the generlization
  performance of the trained model every so often. Also, arbitrary metrics may be computed.

- **Callbacks**: During training and model evaluation, callbacks serve as hooks called at specific
  events in the process. This makes it possible to easily use some tracking framework, perform
  early stopping, or dynamically adjust parameters over the course of the training. Custom
  callbacks can easily be created.

- **GPU Support**: Training and model evaluation is automatically performed on all available GPUs.
  The same code that works for the CPU works for the GPU ... and also for multiple GPUs.

Available Engines
^^^^^^^^^^^^^^^^^

Engines are currently implemented for the following training procedures:

- :class:`pyblaze.nn.MLEEngine`: This is the most central engine as it enables supervised as well as
  unsupservised learning. It can therefore adapt to multiple different problems: classification,
  regression, (variational) autoencoders, ... --- depending on the loss only. In order to simplify
  initialization (as configuration requires toggling some settings), there exist some specialized
  MLE engines. Currently, the only one is :class:`pyblaze.nn.AutoencoderEngine`.

- :class:`pyblaze.nn.WGANEngine`: This engine is specifically designed for training Waserstein GANs.
  This class is required due to the independent training of generator and critic.

Implementing your custom engine is rarely necessary for most common problems. However, when working
on highly customized machine learning models, it might be a good idea. Usually, it is sufficient to
implement the :meth:`train_batch` and :meth:`eval_batch` methods to specify how to perfor training
and evaluation, respectively, for a single batch of data. Consult the documentation of
:class:`pyblaze.nn.Engine` to read about all methods available for override.

.. .. toctree::
..     :glob:
..     :maxdepth: 1
..     :caption: Basics

..     guides/classifier
..     guides/multiprocessing

.. .. toctree::
..     :glob:
..     :maxdepth: 1
..     :caption: Generative Models

..     guides/vae
..     guides/wgan
..     guides/normalizing-flows

.. toctree::
    :glob:
    :maxdepth: 1
    :caption: Package Reference

    modules/*

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
