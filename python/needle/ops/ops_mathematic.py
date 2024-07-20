"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND 
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
        ### BEGIN YOUR SOLUTION
        # numpy.power()
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        # 这里的运算使用的都是needle.Tensor的重载运算符
        return out_grad * self.scalar * input ** (self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # numpy.divide()
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad / b, out_grad * a / (- b ** 2)
       ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        # 标量在建图时不算在inputs里，而是作为op的属性
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # 注意NDArray中实现的permute和这里的transpose语义上的不同（transpose只交换两个轴）
        new_axes = list(range(len(a.shape)))
        if self.axes is None:
            # 交换最后两个轴
            new_axes[-2], new_axes[-1] = new_axes[-1], new_axes[-2]
        else:
            # 交换两个轴
            new_axes[self.axes[0]], new_axes[self.axes[1]] = new_axes[self.axes[1]], new_axes[self.axes[0]]
        return a.permute(tuple(new_axes))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 再swap一遍就回去了
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        return reshape(out_grad, input_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        output_shape = self.shape
        # numpy的广播策略是将维度先`右`对齐，然后从右往左比较
        # 所以如果维数增加(如1D->2D)，将input_shape缺失的维度从左开始填1与output_shape对齐
        tmp_shape = [1] * (len(output_shape) - len(input_shape)) + list(input_shape)
        dele_shape = []
        for i in range(len(output_shape)):
            # 检查每一维是否被扩展
            if output_shape[i] != tmp_shape[i]:
                dele_shape.append(i)
        # 将所有被扩展的维度通过sum压缩回去
        # tensor sum的实现中，numpy.sum如果不设置keepdims, 会去掉值为1的dim，结果形状出现问题.
        # 如(1,3) broadcast_to (3,3) sum 却得到 (3,)
        # 所以最后还需要reshape成输入的形状，以避免上述问题
        return out_grad.sum(tuple(dele_shape)).reshape(input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    # axes: tuple or None
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return a.sum(axis=None)
        elif isinstance(self.axes, int) or (isinstance(self.axes, (list, tuple)) and len(self.axes) == 1):
            return a.sum(self.axes)
        else:
            # NDArray实现的sum(CPU/CUDA) 只能reduce sum一个维度
            # 由于Broadcast的gradient计算可能需要sum多个维度，所以要实现sum多个维度
            for axis in reversed(sorted(self.axes)):
                a = a.sum(axis=axis)
            return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        # numpy.sum如果不设置keepdims, 会去掉值为1的dim，所以手动把keepdim时的形状求出并reshape(sum的特点是shape[axes]值都变为1)，保证broadcast时形状正确
        shape_keepdims = list(input_shape)
        # None means sum along all dim
        if self.axes is None:
            shape_keepdims = [1] * len(input_shape)
        elif isinstance(self.axes, tuple):
            for index in self.axes:
                shape_keepdims[index] = 1
        elif isinstance(self.axes, int):
                shape_keepdims[self.axes] = 1
        else:
            raise ValueError("Unsupported axes type, must be int, tuple or None!")
        return out_grad.reshape(tuple(shape_keepdims)).broadcast_to(input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # 不能用arrayapi.matmul, 因为ndarray.py中没有定义这个函数（应该是作者忘了，因为log之类的都有函数），只重写了NDArray的matmul方法
        return a @ b
        ### END YOUR SOLUTION

    # 有些我自己补充的测试用例过不了，懒得扣细节了，需要详细判断a_value和b_value哪些维度被广播了
    # 想法：先补1把a_value和b_value的shape长度搞一样，然后zip到一起进行循环并判断，除掉最后两个维度，哪个值小哪个输入(a_value or b_value)的该维度广播
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a_value, b_value = node.inputs
        if len(a_value.shape) == len(b_value.shape):
            return out_grad.matmul(b_value.transpose()), a_value.transpose().matmul(out_grad)
        # For inputs with more than 2 dimensions
        # we treat the last two dimensions as being the dimensions of the matrices to multiply, and ‘broadcast’ across the other dimensions.
        elif len(a_value.shape) < len(b_value.shape):
            # 不考虑最后两个维度，导数需要沿广播的维度`sum`以保证形状正确
            axes = range(len(b_value.shape) - len(a_value.shape))
            return out_grad.matmul(b_value.transpose()).sum(axes=tuple(axes)), a_value.transpose().matmul(out_grad)
        else:
            # 不考虑最后两个维度，导数需要沿广播的维度`sum`以保证形状正确
            axes = range(len(a_value.shape) - len(b_value.shape))
            return out_grad.matmul(b_value.transpose()), a_value.transpose().matmul(out_grad).sum(axes=tuple(axes))
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * (a ** -1)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        # use exp method in needle ops, not in numpy
        return out_grad * exp(a)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    # it's acceptable to access the .realize_cached_data() call on the output tensor
    # since the ReLU function is not twice differentiable anyway.
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0].realize_cached_data()
        mask = a > 0
        # 创建Tensor时，需要指定device参数，否则device不同会报错
        return out_grad * Tensor(mask, device=out_grad.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        # -Tensor+scalar会正确调用Tensor类重写的函数
        # 而scalar-Tensor也会调用Tensor类重写的__sub__函数，此时函数的other参数仍然会是scalar！（因为Tensor类编写了__rsub__=__sub__，也不知道为什么这样处理）
        # 结合重写的__sub__函数的实现，会导致结果正好为相反数（Tensor-scalar）
        # https://stackoverflow.com/questions/35736193/what-is-a-typical-instance-of-using-rsub-method-in-python
        return out_grad * (-tanh(a) ** 2 + 1)
        ### END YOUR SOLUTION


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

    # 记住compute方法是修改cached_data的，应该直接返回NDArray
    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # stack一定会扩展维度，和torch.cat不一样
        # https://pytorch.org/docs/stable/generated/torch.stack.html
        # 总体思路：先求出new_shape，然后把要stack的维度定住，每次填充其他维度
        # 例如args[0].shape=(5, 5), self.axis = 2, len(args) = 3
        # 就把第二维定住，每次填充:new_arr[0:5, 0:5, 0 -> 1 -> 2] = args[0 -> 1 -> 2]
        # 如果self.axis = 1, 每次填充:new_arr[0:5, 0 -> 1 -> 2, 0:5] = args[0 -> 1 -> 2]
        new_shape = list(args[0].shape)
        new_shape.insert(self.axis, len(args))
        new_arr = array_api.empty(shape=new_shape, device=args[0].device)
        idxs = []
        for sh in args[0].shape:
            # slice切片: start, stop, step
            idxs.append(slice(0, sh, 1))

        for i in range(len(args)):
            new_idxs = idxs.copy()
            # 例如args[0].shape=(5, 5), self.axis = 2, len(args) = 3
            # new_idxs:[slice(0, 5, 1), slice(0, 5, 1), 0 -> 1 -> 2]
            new_idxs.insert(self.axis, i)
            new_arr[tuple(new_idxs)] = args[i]

        return new_arr
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, axis=self.axis)
        ### END YOUR SOLUTION


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
        ### BEGIN YOUR SOLUTION
        result = []
        idxs = []
        for index, sh in enumerate(A.shape):
            if index != self.axis:
                idxs.append(slice(0, sh, 1))
        for i in range(A.shape[self.axis]):
            new_idxs = idxs.copy()
            new_idxs.insert(self.axis, i)
            # 把值为1的维度(split的维度)去掉
            # getitem以后得到的结果是非连续的，进行后续操作之前需要先compact
            result.append(A[tuple(new_idxs)].compact().sum(self.axis))
        return tuple(result)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, axis=self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        for axis in self.axes:
            new_shape[axis] += self.dilation * new_shape[axis]
        new_arr = array_api.full(shape=new_shape, device=a.device, fill_value=0)
        slices = []
        for index, value in enumerate(a.shape):
            # 该维度不膨胀
            if index not in self.axes:
                slices.append(slice(0, a.shape[index], 1))
            else:
                slices.append(slice(0, new_shape[index], self.dilation + 1))
        new_arr[tuple(slices)] = a
        return new_arr
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        slices = []
        for index, value in enumerate(a.shape):
            if index not in self.axes:
                slices.append(slice(0, a.shape[index], 1))
            else:
                slices.append(slice(0, a.shape[index], self.dilation + 1))
        return a[tuple(slices)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        # simply apply your padding function to the spatial dimensions (i.e., axes 1 and 2)
        axes = [(0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)]
        A_padded = A.pad(axes=axes)
        N, H, W, C = A_padded.shape
        # c_in == C
        k, _, _, c_out = B.shape
        N_stride, H_stride, W_stride, C_stride = A_padded.strides
        # assume stride=1 first
        # A_ = A_padded.as_strided(shape=(N, H-k+1, W-k+1, k, k, C),
        #                          strides=(N_stride, H_stride, W_stride, H_stride, W_stride, C_stride))
        # A_matrix = A_.compact().reshape(new_shape=(N * (H-k+1) * (W-k+1), k * k * C))

        # 应确保能整除
        # H和W保证一定相等，所以只考虑H其实也可以
        # H_after_conv = W_after_conv = (H - k) // self.stride + 1
        H_after_conv = (H - k) // self.stride + 1
        W_after_conv = (W - k) // self.stride + 1
        A_ = A_padded.as_strided(shape=(N, H_after_conv, W_after_conv, k, k, C),
                                 strides=(N_stride, H_stride * self.stride, W_stride * self.stride, H_stride, W_stride, C_stride))
        A_matrix = A_.compact().reshape(new_shape=(N * H_after_conv * W_after_conv, k * k * C))
        # 权重矩阵可能也不是连续的，这在gradient中就有体现（transposed_fliped_weight不是连续的）
        B_matrix = B.compact().reshape(new_shape=(k * k * C, c_out))
        matrix_res = A_matrix @ B_matrix
        return matrix_res.reshape(new_shape=(N, H_after_conv, W_after_conv, c_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # X, grad_X: NHWC_in, weight, grad_weight: KKIO, out_grad: NH'W'C_out
        X, weight = node.inputs
        # N, H, W, C = X.shape
        # _, H_after_conv, W_after_conv, _ = out_grad.shape
        k, _, _, c_out = weight.shape

        # 把stride>1时的out_grad膨胀到和stride=1时一样的维度，后面都和stride=1一样处理即可
        if self.stride > 1:
            out_grad = dilate(out_grad, (1, 2), self.stride - 1)

        # flipped over both the kernel dimensions(K, K, I, O)
        fliped_weight = flip(weight, axes=(0, 1))
        # whatever makes the dimensions work out is typically right
        # 把结果形状搞对，一般都是对的
        # 要把grad_Z的形状搞对(NHWC_in), weight的形状应该为KKOI
        transposed_fliped_weight = transpose(fliped_weight, axes=(-1, -2))
        # out_grad则必须进行padding
        # 想要(H'+2P-K)//S+1=H，S取1，且有H' = (H + 2*self.padding - k) // self.stride + 1, 考虑self.stride=1的情况，解出P
        # H和W保证一定相等，所以只考虑H即可
        padding = k - 1 - self.padding
        grad_X = conv(out_grad, transposed_fliped_weight, stride=1, padding=padding)
        # 要把grad_W的形状搞对(KKIO), X的形状应该为C_inHWN, out_grad的形状应该为H'W'NC_out, conv后的形状应该为C_inH''W''C_out
        transposed_X = transpose(X, axes=(0, -1))
        transposed_out_grad = transpose(transpose(out_grad, axes=(0, 1)), axes=(1, 2))
        # 想要H''=(H+2P-H')//S+1=K，S取1，且有H' = (H + 2*self.padding - k) // self.stride + 1, 考虑self.stride=1的情况，解出P
        padding = self.padding
        conv_res = conv(transposed_X, transposed_out_grad, stride=1, padding=padding)
        # 再把conv的结果permute成KKIO
        grad_weight = transpose(transpose(conv_res, axes=(0, 1)), axes=(1, 2))
        return grad_X, grad_weight
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)