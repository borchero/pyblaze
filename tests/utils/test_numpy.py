import unittest
import numpy as np
from pyblaze.utils import numpy as xnp

class TestNumpy(unittest.TestCase):
    """
    Test case to test some numpy utils.
    """

    def test_find_contexts(self):
        """
        Unit test to find 2D contexts.
        """
        # Without pivot
        seed = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ])
        result = xnp.find_contexts(seed, 2)

        expected = np.array([
            [[1, 2], [2, 3], [3, 4]],
            [[5, 6], [6, 7], [7, 8]]
        ])
        self.assertTrue((expected == result).all())

        # With pivot
        result = xnp.find_contexts(seed, 3, exclude_pivot=True)

        expected_contexts = np.array([
            [[1, 3], [2, 4]],
            [[5, 7], [6, 8]]
        ])
        expected_targets = np.array([
            [2, 3],
            [6, 7]
        ])

        self.assertTrue((expected_contexts == result[0]).all())
        self.assertTrue((expected_targets == result[1]).all())

    def test_find_contexts_1d(self):
        """
        Unit test to find 1D contexts.
        """
        seed = np.array([
            1, 2, 3, 4, 5, 6
        ])
        expected = np.array([
            [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]
        ])
        result = xnp.find_contexts(seed, 2)

        self.assertTrue((expected == result).all())

    def test_find_contexts_nd(self):
        """
        Unit test to find nD contexts.
        """
        seed = np.array([
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8]
            ],
            [
                [9, 10, 11, 12],
                [13, 14, 15, 16]
            ]
        ])
        expected = np.array([
            [
                [[1, 2, 3], [2, 3, 4]],
                [[5, 6, 7], [6, 7, 8]]
            ],
            [
                [[9, 10, 11], [10, 11, 12]],
                [[13, 14, 15], [14, 15, 16]]
            ]
        ])
        result = xnp.find_contexts(seed, 3)

        self.assertTrue((expected == result).all())


if __name__ == '__main__':
    unittest.main()
