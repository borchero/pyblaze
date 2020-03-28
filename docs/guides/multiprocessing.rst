Multiprocessing Module
======================

When working with independent data, spreading computations across multiple cores often provides an easy way to (linearly) increase a program's throughput.

One of the most common forms of parallelism is to split loops across multiple threads. Unfortunately, Python does not support such parallelism easily, especially if computations are CPU-bound. Using simple threads is often not an option due to the GIL and working with processes is often tedious. This is particularly true when working with PyTorch tensors as they have to be passed between processes over queues.

The `multiprocessing` module of PyBlaze aims to make it possible to speed up computations easily, providing a high-level interface.

Vectorization
-------------

PyBlaze refers to vectorization as the process of parallelizing for-loops of the following form:

.. code-block:: python

    result = []
    for item in iterable:
        result.append(map(item))

PyBlaze's class providing this functionality is the `Vectorizer` class in the `multiprocessing` module. In the background, the vectorizer handles everything such as creating processes, ensuring their shutdown, passing items and results between processes. Due to the class's simplicity, it can often be used as a drop-in replacement for existing for-loops which not only reduces runtime but enhances readability.

Example Program
^^^^^^^^^^^^^^^

Consider, for example, an array `text` of strings which you want to tokenize according to a complex function `tokenize`. The function takes as input a single string and returns its tokenization. 

.. code-block:: python

    texts = [
        ''.join(np.random.choice(['a', 'b', 'c', 'd', 'e', ' '], size=(20,)))
        for _ in range(100)
    ]

    def tokenize(text):
        time.sleep(0.01)
        return text.split()

The sequential implementation is very easy, however, not particularly efficient:

.. code-block:: python

    def sequential(texts):
        return [tokenize(t) for t in texts]

The `Vectorizer` can easily be used to compute the tokenizations of all texts in parallel:

.. code-block:: python

    import pyblaze.multiprocessing as xmp

    def parallel(texts):
        vectorizer = xmp.Vectorizer(tokenize, num_workers=4)
        return vectorizer.process(texts)

We can now compare the runtime of the sequential and vectorized implementation:

>>> %timeit sequential(texts)
1.17 s ± 9.37 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

>>> %timeit parallel(texts)
295 ms ± 2.03 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

In this case, the speedup on four cores is 3.97. Hence, the vectorized implementation achieves an almost linear speedup by distributing work across processes.

Advanced Features
^^^^^^^^^^^^^^^^^

The `Vectorizer` class also provides some more advanced features, such as initializing workers. Consult the class's docs for more information on that.