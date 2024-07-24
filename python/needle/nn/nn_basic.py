"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
            self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype)
        )
        self.bias = Parameter(
            ops.transpose(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype)).data
        ) if bias else None

    def forward(self, X: Tensor) -> Tensor:
        x = ops.matmul(X, self.weight)
        if self.bias is not None:
            x = x + ops.broadcast_to(self.bias, x.shape)
        return x


class Flatten(Module):
    def forward(self, X) -> Tensor:
        shape_product = 1
        for dim in X.shape[1:]:
            shape_product *= dim
        return ops.reshape(X, (X.shape[0], shape_product))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        Iy = init.one_hot(logits.shape[1], y, device=logits.device, dtype=logits.dtype)
        return (ops.summation(ops.logsumexp(logits, axes=(1,)) - ops.summation(logits * Iy, axes=(1,))) /
                Tensor(logits.shape[0], dtype=logits.dtype, device=logits.device, requires_grad=False))


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        mean = ops.summation(x, axes=(0,)) / x.shape[0] if self.training else self.running_mean
        mean_broadcasted = ops.implicit_broadcast(mean, x.shape, False)
        variance = ops.summation((x - mean_broadcasted) ** 2, axes=(0,)) / x.shape[
            0] if self.training else self.running_var

        if self.training:
            self.running_mean = (self.momentum * mean + (1 - self.momentum) * self.running_mean).data
            self.running_var = (self.momentum * variance + (1 - self.momentum) * self.running_var).data

        x_hat = (x - mean_broadcasted) / ops.implicit_broadcast((variance + self.eps) ** (1 / 2), x.shape, False)
        return x_hat * ops.implicit_broadcast(self.weight, x.shape, False) + ops.implicit_broadcast(self.bias, x.shape,
                                                                                                    False)


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2, 3)).transpose((1, 2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == self.dim
        expectation = ops.implicit_broadcast(ops.summation(x, axes=(1,)) / self.dim, x.shape)
        variance = ops.summation((x - expectation) ** 2, axes=(1,)) / self.dim
        x_hat = (x - expectation) / ops.implicit_broadcast((variance + self.eps) ** (1 / 2), x.shape)
        return x_hat * ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)),
                                        (x.shape[0], self.dim)) + ops.broadcast_to(
            ops.reshape(self.bias, (1, self.dim)), (x.shape[0], self.dim))


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            bernoulli = np.random.binomial(1, self.p, x.shape)
            bernoulli = (np.negative(bernoulli) + 1) / (1 - self.p)
            '''Notice: Convert it to tensor to ensure the same dtype. Numpy array should be converted to tensor first'''
            return x * Tensor(bernoulli, dtype=x.dtype, requires_grad=False)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)
