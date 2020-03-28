import unittest
import time
import torch
import pyblaze.multiprocessing as xmp

class TestVectorizer(unittest.TestCase):
    """
    Test case to test the Vectorize implementation.
    """

    def setUp(self):
        self.count = None

    def test_on_main_thread(self):
        """
        Unit test to test vectorizer on main thread.
        """
        tensor = torch.arange(100)
        vectorizer = xmp.Vectorizer(divide_by_two, num_workers=0)
        result = vectorizer.process(tensor)
        self.assertTrue(all(torch.as_tensor(result) == tensor / 2))

    def test_batching(self):
        """
        Unit test to test vectorizer with fixed-size input.
        """
        tensor = torch.arange(100)
        vectorizer = xmp.Vectorizer(divide_by_two, num_workers=4)
        result = vectorizer.process(tensor)
        self.assertTrue(all(torch.as_tensor(result) == tensor / 2))

    def test_iterating(self):
        """
        Unit test to test vectorizer with iterable input.
        """
        tensor = torch.arange(100)
        vectorizer = xmp.Vectorizer(divide_by_two, num_workers=4)
        result = vectorizer.process(iter(tensor))
        self.assertTrue(all(torch.as_tensor(result) == tensor / 2))

    def test_worker_init(self):
        """
        Unit test to test vectorizer with worker initialization.
        """
        tensor = torch.arange(100)
        vectorizer = xmp.Vectorizer(
            times, worker_init=factor_from_rank, num_workers=2
        )
        result = vectorizer.process(tensor)
        expected = tensor
        expected[:50] *= 2
        expected[50:] *= 4
        self.assertTrue(all(torch.as_tensor(result) == expected))

    def test_callback(self):
        """
        Unit test to test vectorizer callbacks.
        """
        tensor = torch.arange(100)
        self.count = 0
        def callback():
            self.count += 1
        vectorizer = xmp.Vectorizer(
            divide_by_two, callback_func=callback, num_workers=4
        )
        vectorizer.process(tensor)
        self.assertEqual(self.count, 100)

    def test_speedup(self):
        """
        Unit test to test vectorizer speedup.
        """
        tensor = torch.arange(100)
        seq_vectorizer = xmp.Vectorizer(divide_by_two_slow, num_workers=0)
        par_vectorizer = xmp.Vectorizer(divide_by_two_slow, num_workers=4)

        tic = time.time()
        seq_vectorizer.process(tensor)
        seq_time = time.time() - tic

        tic = time.time()
        par_vectorizer.process(tensor)
        par_time = time.time() - tic

        speedup = seq_time / par_time
        self.assertTrue(
            par_time * 3.5 < seq_time,
            msg=f"Speedup must be at least 3.5 but is {speedup:.2f}."
        )

# pylint: disable=missing-function-docstring
def divide_by_two(x):
    return x / 2

def factor_from_rank(rank):
    return (rank + 1) * 2

def times(x, f):
    return x * f

def divide_by_two_slow(x):
    time.sleep(0.05)
    return x / 2


if __name__ == '__main__':
    unittest.main()
