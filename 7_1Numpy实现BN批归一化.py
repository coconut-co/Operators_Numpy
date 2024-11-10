# 批归一化，均值（mini_batch mean）,方差（mini_batch_variance）,标准化（normalize），缩放和平移（scale，shift）

import numpy as np

class Bach_Nornalization:
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        # 初始化参数 gamma 和 beta，初始值为1和0
        self.gamma = np.ones((num_features,))
        self.beta = np.zeros((num_features,))
        self.momentum = momentum
        self.eps = eps
        
        # 初始化全局均值和方差，用于测试时的推理（inference）
        self.running_mean = np.zeros((num_features,))
        self.running_var = np.ones((num_features,))

    def forward(self, x, train=True):
        if train:
            batch_mean = np.mean(X, axis=0)
            batch_var = np.var(X, axis=0)

            # 更新全局均值和方差，当前批次的均值只对全局均值做出一小部分贡献
            self.running_mean = (self.momentum * self.running_mean) + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            self.x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.eps)
        else:
            # 在推理阶段，使用全局的均值和方差
            self.x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        y = self.gamma * self.x_normalized + self.beta
        return y

X = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0],
              [7.0, 8.0, 9.0]])
bn = Bach_Nornalization(num_features=3)
result = bn.forward(X, train=True)
print(result)