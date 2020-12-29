from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.jit as jit

class StackedLSTM(nn.Module):
    """
    The stacked LSTM is an extension to PyTorch's native LSTM allowing stacked LSTMs with different
    hidden dimensions being stacked. Furthermore, it allows using an LSTM on a GPU without cuDNN.
    This is useful when higher-order gradients are required. In all other cases, it is best to use
    PyTorch's builtin LSTM.
    """

    batch_first: jit.Final[bool]

    def __init__(self, input_size, hidden_sizes, bias=True, batch_first=False, cudnn=True):
        """
        Initializes a new stacked LSTM according to the given parameters.

        Parameters
        ----------
        input_size: int
            The dimension of the sequence's elements.
        hidden_sizes: list of int
            The dimensions of the stacked LSTM's layers.
        bias: bool, default: True
            Whether to use biases in the LSTM.
        batch_first: bool, default: False
            Whether the batch or the sequence can be found in the first
            dimension.
        cudnn: bool, default: True
            Whether to use PyTorch's LSTM implementation which uses cuDNN on Nvidia GPUs. You
            usually don't want to change the default value, however, PyTorch's default
            implementation does not allow higher-order gradients of the LSTMCell as of version
            1.1.0. When this value is set to False, we therefore use a (slower) implementation
            of a LSTM cell which allows higher-order gradients.
        """
        super().__init__()
        self.batch_first = batch_first
        self.stacked_cell = StackedLSTMCell(input_size, hidden_sizes, bias=bias, cudnn=cudnn)

    def forward(self, inputs: torch.Tensor,
                initial_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                return_sequence: bool = True):
        """
        Computes the forward pass through the stacked LSTM.

        Parameters
        ----------
        inputs: torch.Tensor [S, B, N]
            The inputs fed to the LSTM one after the other. Sequence length S, batch size B, and
            input size N. If `batch_first` is set to True, the first and second dimension should
            be swapped.
        initial_states: list of tuple of (torch.Tensor [H_i], torch.Tensor [H_i]), default: None
            The initial states for all LSTM layers. The length of the list must match the number of
            layers in the LSTM, the sizes of the states must match the hidden sizes of the LSTM
            layers. If None is given, the initial states are defaulted to all zeros.
        return_sequence: bool, default: True
            Whether to return all outputs from the last LSTM layer or only the last one.

        Returns
        -------
        torch.Tensor [S, B, K] or torch.Tensor [B, K]
            Depending on whether sequences are returned, either all outputs or only the output from
            the last cell are returned. If the stacked LSTM was initialized with `batch_first`,
            the first and second dimension are swapped when sequences are returned.
        """
        if self.batch_first:
            inputs = inputs.transpose(1, 0)

        sequence_length = inputs.size(0)

        # Initialize the state to empty vectors is needed for jit to properly
        # compile the function
        if initial_states is None:
            states = [(torch.empty(0), torch.empty(0))]
        else:
            states = initial_states

        # Iterate over sequence
        outputs = []
        for n in range(sequence_length):
            output, states = self.stacked_cell(
                inputs[n], None if states[0][0].size(0) == 0 else states
            )
            if return_sequence or n == sequence_length - 1:
                outputs.append(output)

        if return_sequence:
            result = torch.stack(outputs)
            if self.batch_first:
                # set batch first, sequence length second
                result = result.transpose(1, 0)
            return result
        return outputs[0]


class StackedLSTMCell(nn.Module):
    """
    Actually, the StackedLSTMCell can easily be constructed from existing modules, however, a bug
    in PyTorch's JIT compiler prevents implementing anything where a stacked LSTM is used within a
    loop (see the following issue: https://github.com/pytorch/pytorch/issues/18143). Hence, this
    class provides a single time step for a stacked LSTM.
    """

    cells: jit.Final[int]
    num_stacked: jit.Final[int]

    def __init__(self, input_size, hidden_sizes, bias=True, cudnn=True):
        """
        Initializes a new stacked LSTM cell.

        Parameters
        ----------
        input_size: int
            The dimension of the input variables.
        hidden_sizes: list of int
            The hidden dimension of the stacked LSTMs.
        bias: bool, default: True
            Whether to use a bias term for the LSTM implementation.
        cudnn: bool, default: True
            Whether to not use cuDNN. In almost all cases, you don't want to set this value to
            false, however, you will need to change it if you want to compute higher-order
            derivatives of a network with a stacked LSTM cell.
        """
        super().__init__()

        self.num_stacked = len(hidden_sizes)

        cell_class = nn.LSTMCell if cudnn else _LSTMCell

        cells = []
        dims = zip([input_size] + hidden_sizes, hidden_sizes)
        for in_dim, out_dim in dims:
            cells.append(cell_class(in_dim, out_dim, bias=bias))
        self.cells = nn.ModuleList(cells)

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor,
                initial_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None):
        """
        Computes the new hidden states and cell states for each stacked cell.

        Parameters
        ----------
        x: torch.Tensor [B, N]
            The input with batch size B and dimension N.
        states: list of tuple of (torch.Tensor [B, D], torch.Tensor [B, D]), default: None
            The states for each of the cells where each state is expected to have a size with batch
            size B and (respective) hidden dimension D.

        Returns
        -------
        torch.Tensor [B, D]
            The output, i.e. the hidden state of the deepest cell. Only given for convenience as it
            can be extracted from the other return value.
        list of tuple of (torch.Tensor [B, D], torch.Tensor [B, D])
            The new hidden states and cell states for all cells.
        """
        if initial_states is None:
            # JIT Compatibility
            states = [
                (torch.empty(0), torch.empty(0))
                for _ in range(self.num_stacked)
            ]
        else:
            states = initial_states

        i = 0
        for cell in self.cells:
            x, next_cell = cell(
                x, None if states[i][0].size(0) == 0 else states[i]
            )
            states[i] = (x, next_cell)
            i += 1

        return x, states


class _LSTMCell(nn.Module):
    """
    LSTMCell which does not have cuDNN support but allows for higher-order gradients.
    Consult PyTorch's LSTMCell for documentation on the class's initialization parameters and how
    to call it.
    """

    hidden_size: jit.Final[int]
    has_bias: jit.Final[bool]

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()

        self.hidden_size = hidden_size

        self.input_weight = nn.Parameter(
            torch.FloatTensor(input_size, 4 * hidden_size)
        )
        self.hidden_weight = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
            self.has_bias = True
        else:
            self.has_bias = False

        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the parameters of the model.
        """
        sqrt_hidden = np.sqrt(1 / self.hidden_size)
        init_from = (-sqrt_hidden, sqrt_hidden)
        for p in self.parameters():
            nn.init.uniform_(p, *init_from)

    # pylint: disable=arguments-differ,missing-function-docstring
    def forward(self, x_in: torch.Tensor,
                state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):

        if state is None:
            size = (x_in.size(0), self.hidden_size)
            hidden_state = torch.zeros(
                *size, dtype=torch.float, device=x_in.device
            )
            cell_state = torch.zeros(
                *size, dtype=torch.float, device=x_in.device
            )
        else:
            hidden_state, cell_state = state

        # 1) Perform matrix multiplications for input and last hidden state
        if self.has_bias:
            x = torch.addmm(self.bias, x_in, self.input_weight)
            h = torch.addmm(self.bias, hidden_state, self.hidden_weight)
        else:
            x = x_in.matmul(self.input_weight)
            h = hidden_state.matmul(self.hidden_weight)

        forget_gate_x, input_gate_x_1, input_gate_x_2, output_gate_x = \
            x.split(self.hidden_size, dim=1)
        forget_gate_h, input_gate_h_1, input_gate_h_2, output_gate_h = \
            h.split(self.hidden_size, dim=1)

        # 2) Forget gate
        forget_gate = torch.sigmoid(forget_gate_x + forget_gate_h)

        # 3) Input gate
        input_gate_1 = torch.sigmoid(input_gate_x_1 + input_gate_h_1)
        input_gate_2 = torch.tanh(input_gate_x_2 + input_gate_h_2)
        input_gate = forget_gate * cell_state + input_gate_1 * input_gate_2

        # 4) Output gate
        output_gate_1 = torch.sigmoid(output_gate_x + output_gate_h)
        output_gate = output_gate_1 * torch.tanh(input_gate)

        return output_gate, input_gate
