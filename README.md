# PyBlaze

![PyPi](https://img.shields.io/pypi/v/pyblaze?label=version)

PyBlaze is a high-level library for large-scale machine learning in [PyTorch](https://pytorch.org). It is engineered to cut obsolete boilerplate code while preserving the flexibility of PyTorch to create just about any deep learning model.

## Features

Generally, PyBlaze provides an object-oriented approach to extend PyTorch's API. The core design objective is to provide an API both as simple and as extensible as possible. PyBlaze's features include the following:

* Training and prediction loops with minimal code required and callback support.
* Out-of-the-box multi-GPU support where not a single additional line of code is required.
* Intuitive multiprocessing by providing easy for-loop vectorization.
* Modules and functions missing in PyTorch.

Currently, PyBlaze only provides means for running training/inference on a single machine. In case this is insufficient, you might be better off using PyTorch's `distributed` package directly.

It must be emphasized that PyBlaze is not meant to be a wrapper for PyTorch as Keras is for TensorFlow - it only provides *extensions*.

## Installation

PyBlaze is available on PyPi and can simply be installed as follows:

```bash
pip install pyblaze
```

## Quickstart

An introduction to PyBlaze is given as a tutorial training an image classifier. It can be found in the documentation's [guide section](https://pyblaze.borchero.com/guides/supervised.html).

## License

PyBlaze is licensed under the [MIT License](LICENSE).
