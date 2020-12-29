# PyBlaze

![PyPi](https://img.shields.io/pypi/v/pyblaze?label=version)
![License](https://img.shields.io/github/license/borchero/pyblaze?label=license)

PyBlaze is an unobtrusive, high-level library for large-scale machine and deep learning in
[PyTorch](https://pytorch.org). It is engineered to cut obsolete boilerplate code while preserving
the flexibility of PyTorch to create just about any deep learning model.

## Quickstart

Plenty of tutorials are available in the [official documentation](https://pyblaze.borchero.com/).
The most basic tutorial builds a
[classifier for CIFAR10](https://pyblaze.borchero.com/examples/classifier.html).

### Installation

PyBlaze is available on PyPi and can simply be installed as follows:

```bash
pip install pyblaze
```

## Library Design

PyBlaze revolves around the concept of an **engine**. An engine is a powerful abstraction for
combining a model's definition with the algorithm required to optimize its parameters according to
some data. Engines provided by PyBlaze are focused on generalization: while the engine encapsulates
the optimization algorithm, the user must explicitly define the optimization objective (usually the
loss function).

However, engines go far beyond implementing the optimization algorithm. Specifically, they further
provide the following features:

- **Evaluation**: During training, validation data can be used to evaluate the generalization
  performance of the trained model every so often. Also, arbitrary metrics may be computed.

- **Callbacks**: During training and model evaluation, callbacks serve as hooks called at specific
  events in the process. This makes it possible to easily use some tracking framework, perform
  early stopping, or dynamically adjust parameters over the course of the training. Custom
  callbacks can easily be created.

- **GPU Support**: Training and model evaluation is automatically performed on all available GPUs.
  The same code that works for the CPU works for the GPU ... and also for multiple GPUs.

### Available Engines

Engines are currently implemented for the following training procedures:

- `pyblaze.nn.MLEEngine`: This is the most central engine as it enables supervised as well as
  unsupervised learning. It can therefore adapt to multiple different problems: classification,
  regression, (variational) autoencoders, ..., depending on the loss only. In order to simplify
  initialization (as configuration requires toggling some settings), there exist some specialized
  MLE engines. Currently, the only one is `pyblaze.nn.AutoencoderEngine`.

- `pyblaze.nn.WGANEngine`: This engine is specifically designed for training Wasserstein GANs.
  This class is required due to the independent training of generator and critic.

Implementing your custom engine is rarely necessary for most common problems. However, when working
on highly customized machine learning models, it might be a good idea. Usually, it is sufficient to
implement the `train_batch` and `eval_batch` methods to specify how to perform training and
evaluation, respectively, for a single batch of data. Consult the documentation of
`pyblaze.nn.Engine` to read about all methods available for override.

## License

PyBlaze is licensed under the [MIT License](LICENSE).
