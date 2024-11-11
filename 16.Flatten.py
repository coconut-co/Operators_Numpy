import numpy as np

class Flatten():
    def __init__(self, x):
        self.x = x
        self.x_shape = x.shape

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        # ravel() 将多维数组展成一维数组
        # [N, C, H, W] -> [N*C*H*W] -> [N, C*H*W]
        output = x.ravel().reshape(self.x_shape[0], -1)
        return output

    def backward(self, d_out):
        d_x = d_out.reshape(self.x_shape)
        return d_x

if __name__ == "__main__":
    x = np.random.rand(2, 3, 4, 4)

    flatten = Flatten(x)
    out = flatten(x)
    print(out.shape)

    d_out = np.random.rand(*out.shape)
    reconstructed_input = flatten.backward(d_out)
    print(reconstructed_input.shape)