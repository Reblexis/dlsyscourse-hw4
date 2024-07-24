from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from .. import backend_ndarray as array_api


class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, axis=self.axes, keepdims=False)
        max_Z_keepdim = array_api.max(Z, axis=self.axes, keepdims=True)
        subbed = Z - array_api.broadcast_to(max_Z_keepdim, Z.shape)
        return array_api.log(array_api.sum(array_api.exp(subbed), axis=self.axes, keepdims=False)) + max_Z
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]

        # whats the new shape after applying maximum with keepdims=true at self.axis
        new_shape = tuple([1 if i in self.axes else Z.shape[i] for i in range(len(Z.shape))])
        exp_t = exp(Z - broadcast_to(reshape(maximum(Z, axes=self.axes), new_shape), Z.shape))
        sum_t = summation(exp_t, axes=self.axes)

        g1 = out_grad / sum_t

        if not self.axes:
            g2 = broadcast_to(g1, exp_t.shape)
        else:
            exp_t_shape = list(exp_t.shape)
            for i in self.axes:
                exp_t_shape[i] = 1
            exp_t_shape = tuple(exp_t_shape)
            g2 = broadcast_to(g1.reshape(exp_t_shape), exp_t.shape)

        return g2 * exp_t
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
