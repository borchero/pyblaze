import unittest
import torch
import torch.nn as nn
from pyblaze.nn.modules import MADE

class TestMADE(unittest.TestCase):
    """
    Test case to check MADE implementation.
    """

    def test_grads(self):
        """
        Unit test to test gradients of MADE when using sequential initialization.
        """
        input_size = 8

        sizes = [
            [input_size, 16, input_size],
            [input_size, 16, input_size * 2],
            [input_size, 16, 16, input_size],
            [input_size, 16, 16, input_size * 2],
            [input_size, 16, 32, 48, input_size * 2]
        ]

        for size in sizes:
            model = MADE(*size, activation=nn.Tanh())
            x = torch.ones(1, input_size, requires_grad=True)
            for d in range(input_size):
                y = model(x)
                loss = y[0, d]
                loss.backward()
                self.assertTrue(torch.all(x.grad[0, :d] != 0))
                self.assertTrue(torch.all(x.grad[0, d:] == 0))
                model.zero_grad()


if __name__ == '__main__':
    unittest.main()
