"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from apps.models import *
import time

device = ndl.cuda()

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """

    project_path = "/content/drive/MyDrive/10714/hw0/"
    with gzip.open(f"{project_path}{image_filename}", 'rb') as f:
        magic_number = f.read(4)
        items_num = struct.unpack('>I', f.read(4))[0]
        rows_num = struct.unpack('>I', f.read(4))[0]
        columns_num = struct.unpack('>I', f.read(4))[0]

        X = np.zeros((items_num, rows_num * columns_num), dtype=np.float32)
        for i in range(items_num):
            pixels = f.read(rows_num * columns_num)
            image = np.frombuffer(pixels, dtype=np.uint8)
            image_normalized = image.astype(np.float32) / 255
            X[i] = image_normalized

    with gzip.open(f"{project_path}{label_filename}", 'rb') as f:
        magic_number = f.read(4)
        items_num = struct.unpack('>I', f.read(4))[0]
        y = np.zeros(items_num, dtype=np.uint8)
        for i in range(items_num):
            label_byte = f.read(1)
            y[i] = np.frombuffer(label_byte, dtype=np.uint8)

    return X, y


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    res1 = ndl.log(ndl.summation(ndl.exp(Z), axes=(1,)))
    res2 = ndl.summation(Z * y_one_hot, axes=(1,))
    res3 = res1 - res2
    return ndl.summation(res3) / Z.shape[0]
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    for i in range(int((X.shape[0] + batch - 1) / batch)):
        batch_size = min((i + 1) * batch, X.shape[0]) - i * batch
        logits_size = W2.shape[1]
        X_batch = ndl.Tensor(X[i * batch: i * batch + batch_size])
        y_batch = ndl.Tensor(y[i * batch: i * batch + batch_size])
        Iy = np.zeros((batch_size, logits_size))
        Iy[np.arange(batch_size), y_batch.numpy()] = 1

        loss = softmax_loss(ndl.matmul(ndl.relu(ndl.matmul(X_batch, W1)), W2), ndl.Tensor(Iy))
        loss.backward()

        W1 = ndl.Tensor(W1.detach() - lr * W1.grad.detach())
        W2 = ndl.Tensor(W2.detach() - lr * W2.grad.detach())

    return W1, W2


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)


### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    model.eval() if opt is None else model.train()

    losses = []
    total = 0
    correct = 0

    for X, y in dataloader:
        X = ndl.Tensor(X, device=device)
        y = ndl.Tensor(y, device=device)
        logits = model(X)
        loss = loss_fn(logits, y)
        losses.append(loss.numpy())
        total += y.shape[0]
        correct += np.sum(np.argmax(logits.numpy(), axis=1) == y.numpy())

        if opt is not None:
            loss.backward()
            opt.step()

        print(f"Progress: {total} / {len(dataloader.dataset)}", end="\r")

    return correct / total, float(np.mean(losses))


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
                  lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)

    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn(), opt)
        print(f"Epoch {epoch + 1}: avg_acc={avg_acc:.4f}, avg_loss={avg_loss:.4f}")

    return avg_acc, avg_loss


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn())
    print(f"Test: avg_acc={avg_acc:.4f}, avg_loss={avg_loss:.4f}")

    return avg_acc, avg_loss


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
                      clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    model.eval() if opt is None else model.train()

    losses = []
    total = 0
    correct = 0

    for i in range(data.shape[0] - 1):
        X, y = ndl.data.get_batch(data, i, seq_len, device=device)
        logits, _ = model(X)
        loss = loss_fn(logits, y)
        losses.append(loss.numpy())
        total += y.shape[0]
        correct += np.sum(np.argmax(logits.numpy(), axis=1) == y.numpy())

        if opt is not None:
            loss.backward()
            if clip is not None:
                #ndl.nn.utils.clip_grad_norm_(model.parameters(), clip)
                pass
            opt.step()

        print(f"Progress: {i} / {data.shape[0]-1}, acc={correct / total:.4f}, loss={np.mean(losses):.4f}", end="\r")

    return correct / total, float(np.mean(losses))


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
              lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
              device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)

    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len, loss_fn(), opt, clip, device, dtype)
        print(f"Epoch {epoch + 1}: avg_acc={avg_acc:.4f}, avg_loss={avg_loss:.4f}")

    return avg_acc, avg_loss


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
                 device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len, loss_fn(), device=device, dtype=dtype)
    print(f"Test: avg_acc={avg_acc:.4f}, avg_loss={avg_loss:.4f}")

    return avg_acc, avg_loss


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
