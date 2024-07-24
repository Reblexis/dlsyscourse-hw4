"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {self.params[i]: ndl.zeros_like(self.params[i]) for i in range(len(self.params))}
        self.weight_decay = weight_decay

    def step(self):
        for w in self.params:
            self.u[w] = ndl.Tensor(self.momentum * self.u[w] + (1 - self.momentum) * (w.grad.data + self.weight_decay * w.data), device=w.device, dtype=w.dtype)
            w.data = w.data - self.lr * self.u[w]

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {self.params[i]: ndl.zeros_like(self.params[i]) for i in range(len(self.params))}
        self.v = {self.params[i]: ndl.zeros_like(self.params[i]) for i in range(len(self.params))}

    def step(self):
        self.t += 1
        for w in self.params:
            g = ndl.Tensor(w.grad.data + self.weight_decay * w.data, device=w.device, dtype=w.dtype, requires_grad=False)
            self.m[w] = self.beta1 * self.m[w] + (1 - self.beta1) * g
            self.v[w] = self.beta2 * self.v[w] + (1 - self.beta2) * g ** 2
            m_hat = self.m[w] / (1 - self.beta1 ** self.t)
            v_hat = self.v[w] / (1 - self.beta2 ** self.t)
            w.data = w.data - self.lr * m_hat / (v_hat ** (1/2) + self.eps)