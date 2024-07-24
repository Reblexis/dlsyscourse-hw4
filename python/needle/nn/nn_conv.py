"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # Initialize weights
        self.weight = Parameter(
            init.kaiming_uniform(kernel_size*kernel_size*in_channels, kernel_size*kernel_size*out_channels, shape=(kernel_size, kernel_size, in_channels, out_channels), device=device, dtype=dtype)
        )
        bias_interval = 1.0/(in_channels*kernel_size*kernel_size)
        self.bias = Parameter(
            init.rand(out_channels, low=-bias_interval, high=bias_interval, device=device, dtype=dtype)
        )
        self.padding = kernel_size // 2 # works for only odd kernel sizes

    def forward(self, x: Tensor) -> Tensor:
        x = ops.transpose(ops.transpose(x, (1, 3)), (1, 2))
        x = ops.conv(a=x, b=self.weight, stride=self.stride, padding=self.padding)
        x = ops.transpose(ops.transpose(x, (1, 2)), (1, 3))
        x = x + ops.broadcast_to(ops.reshape(self.bias, (1, self.out_channels, 1, 1)), x.shape)
        return x
