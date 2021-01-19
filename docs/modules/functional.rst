Functional
==========

.. code:: python

    import pyblaze.nn.functional as X

The functional module provides free functions that are missing from PyTorch.

.. contents:: Contents
    :local:
    :depth: 1

General Functions
-----------------

Gumbel
^^^^^^

.. autofunction:: pyblaze.nn.functional.gumbel_softmax

Soft Round
^^^^^^^^^^

.. autofunction:: pyblaze.nn.functional.softround

Probability Distributions
-------------------------

Normal Distribution
^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyblaze.nn.functional.log_prob_standard_normal

Metrics
-------

Accuracy
^^^^^^^^

.. autofunction:: pyblaze.nn.functional.accuracy

Recall
^^^^^^

.. autofunction:: pyblaze.nn.functional.recall

Precision
^^^^^^^^^

.. autofunction:: pyblaze.nn.functional.precision

F1 Score
^^^^^^^^

.. autofunction:: pyblaze.nn.functional.f1_score

ROC-AUC Score
^^^^^^^^^^^^^

.. autofunction:: pyblaze.nn.functional.roc_auc_score

Average Precision
^^^^^^^^^^^^^^^^^

.. autofunction:: pyblaze.nn.functional.average_precision
