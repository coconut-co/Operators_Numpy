import torch
import torch.nn as nn
import numpy as np

class NP_BN:
    def __init__(self, feature_nums, eps=1e-5, momentum=0.8):
        self.gamma = torch.ones((feature_nums,))
        self.beta = torch.zeros((feature_nums,))
        self.eps = eps
        self.momentum = momentum

        self.running_mean = torch.zeros((feature_nums,))
        self.running_var = torch.ones((feature_nums,))

    def forward(self, x, train=True):
        if train:
            batch_mean = torch.mean(x, dim=0)
            batch_var = torch.var(x, dim=0)

            self.x_normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = (self.momentum * self.running_var) + (1 - self.momentum) * batch_var

        else:
            self.x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        output_np = self.gamma * self.x_normalized + self.beta

        return output_np
    
x = torch.randn(10, 5)
bn = NP_BN(5)
output = bn.forward(x)
print(output)

bn_torch = nn.BatchNorm1d(num_features=5, momentum=0.8, eps=1e-5)
with torch.no_grad():
    bn_torch.weight.fill_(1.0)  # gamma
    bn_torch.bias.fill_(0.0)    # beta

output_torch = bn_torch(x).detach().numpy()
print(output_torch)


# 比较输出
print("Difference between custom and PyTorch outputs during training:", np.abs(output - output_torch).max())