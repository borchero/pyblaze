import unittest
import numpy as np
import torch
import sklearn.metrics as m
import pyblaze.nn.functional as X

class TestMetrics(unittest.TestCase):
    """
    Test case to test metrics.
    """

    def test_roc_auc_score(self):
        """
        Unit test to test ROC-AUC score.
        """
        for _ in range(100):
            y_pred = np.random.uniform(size=(100,))
            y_true = np.random.choice(2, size=(100,))

            roc_pred = X.roc_auc_score(
                torch.from_numpy(y_pred), torch.from_numpy(y_true)
            )
            roc_true = m.roc_auc_score(y_true, y_pred)

            self.assertAlmostEqual(roc_pred.numpy(), roc_true)


if __name__ == '__main__':
    unittest.main()
