"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return (1.0 + ops.exp(-x)) ** (-1)


def sigmoid(x: Tensor) -> Tensor:
    return Sigmoid()(x)


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()

        self.use_bias = bias

        init_bound = np.sqrt(1 / hidden_size)
        self.hidden_size = hidden_size
        self.W_ih = Parameter(
            init.rand(input_size, hidden_size, device=device, dtype=dtype, low=-init_bound, high=init_bound))
        self.W_hh = Parameter(
            init.rand(hidden_size, hidden_size, device=device, dtype=dtype, low=-init_bound, high=init_bound))
        self.bias_ih = Parameter(
            init.rand(hidden_size, device=device, dtype=dtype, low=-init_bound, high=init_bound)) if bias else None
        self.bias_hh = Parameter(
            init.rand(hidden_size, device=device, dtype=dtype, low=-init_bound, high=init_bound)) if bias else None

        self.nonlin = ops.tanh if nonlinearity == 'tanh' else ops.relu

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """

        inner_res = X @ self.W_ih
        if h is not None:
            inner_res += h @ self.W_hh

        if self.use_bias:
            inner_res += (self.bias_ih + self.bias_hh).reshape((1, self.hidden_size)).broadcast_to(
                (X.shape[0], self.hidden_size))

        return self.nonlin(inner_res)


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None,
                 dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn_cells = []
        for i in range(self.num_layers):
            self.rnn_cells.append(
                RNNCell(input_size if i == 0 else hidden_size, hidden_size, bias=bias, nonlinearity=nonlinearity,
                        device=device, dtype=dtype))

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        seq_len, bs, input_size = X.shape
        assert input_size == self.input_size

        h_n = ops.split(h0, 0).list() if h0 is not None else [
            init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype) for _ in range(self.num_layers)]
        outputs = ops.split(X, 0).list()

        for i in range(self.num_layers):
            for t in range(seq_len):
                outputs[t] = self.rnn_cells[i](outputs[t], h_n[i])
                h_n[i] = outputs[t]

        return ops.stack(outputs, 0), ops.stack(h_n, 0)


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = bias

        init_bound = np.sqrt(1 / hidden_size)
        self.W_ih = Parameter(
            init.rand(input_size, 4 * hidden_size, device=device, dtype=dtype, low=-init_bound, high=init_bound))
        self.W_hh = Parameter(
            init.rand(hidden_size, 4 * hidden_size, device=device, dtype=dtype, low=-init_bound, high=init_bound))
        self.bias_ih = Parameter(
            init.rand(4 * hidden_size, device=device, dtype=dtype, low=-init_bound, high=init_bound)) if bias else None
        self.bias_hh = Parameter(
            init.rand(4 * hidden_size, device=device, dtype=dtype, low=-init_bound, high=init_bound)) if bias else None

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """

        bs = X.shape[0]
        if h is None:
            h = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)
            c = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)
        else:
            h, c = h

        gates = X @ self.W_ih + h @ self.W_hh
        if self.use_bias:
            gates += (self.bias_ih + self.bias_hh).reshape((1, 4 * self.hidden_size)).broadcast_to(
                (bs, 4 * self.hidden_size))

        gates_split = ops.split(gates.reshape((bs, 4, self.hidden_size)), 1)
        i, f, g, o = [sigmoid(gates_split[i]) if i != 2 else ops.tanh(gates_split[i]) for i in range(4)]

        c = f * c + i * g
        h = o * ops.tanh(c)

        return h, c


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm_cells = []
        for i in range(self.num_layers):
            self.lstm_cells.append(
                LSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias=bias, device=device, dtype=dtype))

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            c_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """

        seq_len, bs, input_size = X.shape
        assert input_size == self.input_size

        h0, c0 = h if h is not None else (None, None)
        h_n = ops.split(h0, 0).list() if h0 is not None else [
            init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype) for _ in range(self.num_layers)]
        c_n = ops.split(c0, 0).list() if c0 is not None else [
            init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype) for _ in range(self.num_layers)]

        outputs = ops.split(X, 0).list()

        for i in range(self.num_layers):
            for t in range(seq_len):
                h_n[i], c_n[i] = self.lstm_cells[i](outputs[t], (h_n[i], c_n[i]))
                outputs[t] = h_n[i]

        return ops.stack(outputs, 0), (ops.stack(h_n, 0), ops.stack(c_n, 0))


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
