import unittest
import numpy as np
import torch
import torch.nn as nn
from pyblaze.nn.modules.lstm import _LSTMCell

class TestLSTM(unittest.TestCase):
    """
    Test case to check LSTM implementation.
    """

    def test_lstm_cell(self):
        """
        Unit test to test LSTM cell.
        """
        nn_lstm = nn.LSTMCell(2, 2, bias=False)
        xnn_lstm = _LSTMCell(2, 2, bias=False)

        for _ in range(50):
            x = torch.rand(2).view(1, -1)
            hidden = torch.rand(2).view(1, -1)
            state = torch.rand(2).view(1, -1)

            val = float(np.random.rand(1))
            for p in nn_lstm.parameters():
                nn.init.constant_(p, val)
            for p in xnn_lstm.parameters():
                nn.init.constant_(p, val)

            with torch.no_grad():
                out_true = nn_lstm(x, (hidden, state))
                out_ours = xnn_lstm(x, (hidden, state))

            self.assertTrue(
                torch.allclose(out_true[0], out_ours[0])
            )
            self.assertTrue(
                torch.allclose(out_true[1], out_ours[1])
            )

if __name__ == '__main__':
    unittest.main()
