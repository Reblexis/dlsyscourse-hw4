"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from .. import backend_ndarray as array_api
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        return out_grad * self.scalar * (node.inputs[0] ** (self.scalar - 1))


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a ** b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
                node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a ** b) * array_api.log(a.data)
        return grad_a, grad_b


def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad / rhs, out_grad * (-lhs / (rhs ** 2))


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return (out_grad / self.scalar,)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # reverses the order of two axes (axis1, axis2), defaults to the last two axes (1 input, axes - tuple)
        if self.axes is None:
            self.axes = (len(a.shape) - 2, len(a.shape) - 1)

        axes_permutation = list(range(len(a.shape)))
        axes_permutation[self.axes[0]] = self.axes[1]
        axes_permutation[self.axes[1]] = self.axes[0]

        return array_api.transpose(a, axes_permutation)

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        return reshape(out_grad, node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        axes_list = []
        j = len(node.inputs[0].shape) - 1
        for i in reversed(range(len(out_grad.shape))):
            if j < 0 or node.inputs[0].shape[j] == 1:
                axes_list.append(i)
            j -= 1

        summ = summation(out_grad, tuple(axes_list))

        return reshape(summ, node.inputs[0].shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


def implicit_broadcast(a, target_shape, backward=True):
    broadcast_shape = list(target_shape)
    volume = numpy.prod(a.shape)
    target_volume = numpy.prod(target_shape)
    the_range = reversed(range(len(target_shape))) if backward else range(len(target_shape))
    for i in the_range:
        if volume >= target_volume:
            break
        target_volume /= target_shape[i]
        broadcast_shape[i] = 1

    if volume > target_volume:
        raise ValueError("Cannot broadcast shapes {} and {}".format(a.shape, target_shape))

    return broadcast_to(reshape(a, tuple(broadcast_shape)), target_shape)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            self.axes = tuple(range(len(a.shape)))
        return array_api.sum(a, self.axes)

    def gradient(self, out_grad, node):
        broadcast_shape = list(node.inputs[0].shape)
        for axis in self.axes:
            broadcast_shape[axis] = 1
        reshaped_grad = reshape(out_grad, tuple(broadcast_shape))
        return broadcast_to(reshaped_grad, node.inputs[0].shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class Maximum(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            self.axes = tuple(range(len(a.shape)))
        return array_api.max(a, self.axes)

    def gradient(self, out_grad, node):
        broadcast_shape = list(node.inputs[0].shape)
        for axis in self.axes:
            broadcast_shape[axis] = 1
        input_realized = node.inputs[0].realize_cached_data()
        reshaped_grad = reshape(out_grad, tuple(broadcast_shape))
        broadcasted_grad = broadcast_to(reshaped_grad, node.inputs[0].shape)
        ones_where_max = (input_realized - array_api.max(
            input_realized, axis=self.axes, keepdims=True
        )) >= 0
        return broadcasted_grad * ones_where_max


def maximum(a, axes=None):
    return Maximum(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        returned_grad_1 = matmul(out_grad, transpose(node.inputs[1]))
        returned_grad_2 = matmul(transpose(node.inputs[0]), out_grad)
        if len(node.inputs[0].shape) < len(out_grad.shape):
            returned_grad_1 = summation(returned_grad_1, tuple(range(len(out_grad.shape) - len(node.inputs[0].shape))))
        if len(node.inputs[1].shape) < len(out_grad.shape):
            returned_grad_2 = summation(returned_grad_2, tuple(range(len(out_grad.shape) - len(node.inputs[1].shape))))
        return reshape(returned_grad_1, node.inputs[0].shape), reshape(returned_grad_2, node.inputs[1].shape)


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return negate(out_grad)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        bigger_than_zero = node.inputs[0].cached_data > 0
        return out_grad * bigger_than_zero


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        return array_api.tanh(a)

    def gradient(self, out_grad, node):
        return out_grad * (1 - tanh(node.inputs[0]) ** 2)


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        stacked = array_api.stack([arg for arg in args], axis=self.axis)
        return stacked

    def gradient(self, out_grad, node):
        return tuple(array_api.split(out_grad, self.axis))


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        return array_api.split(A, self.axis)

    def gradient(self, out_grad, node):
        return array_api.stack(out_grad, axis=self.axis)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



