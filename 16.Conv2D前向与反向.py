import numpy as np
from Img2col import Img2colIndices

class Conv2D():
    def __init__(self, in_channel, out_channel, kernel_size, padding, stride):
        self.in_channel  = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.padding     = padding
        self.stride      = stride

        # 初始化参数
        # kernel: NCHW [out_channel, in_channel, kernel_size, kernel_size]
        self.W = np.random.randn(self.out_channel, self.in_channel, self.kernel_size, self.kernel_size) / np.sqrt(self.out_channel / 2.)
        self.b = np.zeros((self.out_channel, 1))

        self.params = [self.W, self.b]       

    def __call__(self, x):
        self.n_x, self.h_x, self.w_x = x.shape
        self.h_out = (self.h_x + self.padding - self.kernel_size) // self.stride + 1
        self.w_out = (self.w_x + 2 * self.padding - self.kernel_size) // self.stride + 1

        return self.forward(x)