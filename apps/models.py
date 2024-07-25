import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ConvBN(nn.Module):
    def __init__(self, a, b, k, s, device=None, dtype="float32"):
        super().__init__()

        self.conv = nn.Conv(a, b, k, s, device=device, dtype=dtype)
        self.bn = nn.BatchNorm2d(b, device=device, dtype=dtype) # maybe here mistake in incorrect dim
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        self.convbn1 = ConvBN(3, 16, 7, 4, device=device, dtype=dtype)
        self.convbn2 = ConvBN(16, 32, 3, 2, device=device, dtype=dtype)
        self.res1 = nn.Residual(
            nn.Sequential(
                ConvBN(32, 32, 3, 1, device=device, dtype=dtype),
                ConvBN(32, 32, 3, 1, device=device, dtype=dtype)
            )
        )
        self.convbn3 = ConvBN(32, 64, 3, 2, device=device, dtype=dtype)
        self.convbn4 = ConvBN(64, 128, 3, 2, device=device, dtype=dtype) # 32 in channels??
        self.res2 = nn.Residual(
            nn.Sequential(
                ConvBN(128, 128, 3, 1, device=device, dtype=dtype),
                ConvBN(128, 128, 3, 1, device=device, dtype=dtype)
            )
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128, 128, device=device, dtype=dtype)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 10, device=device, dtype=dtype)

    def forward(self, x):
        assert isinstance(x, ndl.Tensor)
        x = self.convbn1(x)
        x = self.convbn2(x)
        x = self.res1(x)
        x = self.convbn3(x)
        x = self.convbn4(x)
        x = self.res2(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        if seq_model == 'rnn':
            self.seq_model = nn.RNN(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        elif seq_model == 'lstm':
            self.seq_model = nn.LSTM(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)

        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)


    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        seq_len, bs = x.shape

        x = self.embedding(x)
        out, h = self.seq_model(x, h)
        out = self.linear(out.reshape((seq_len*bs, self.hidden_size)))
        return out, h


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)