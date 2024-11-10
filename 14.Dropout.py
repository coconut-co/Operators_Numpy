import numpy as np

# 在训练过程中随机“丢弃”一部分神经元（即将它们的输出设为零），来减少模型对特定神经元的依赖,增强模型的泛化能力
# 在训练时会以一定的概率随机选择神经元丢弃，测试时会使用所有神经元的输出
class Dropout:
    def __init__(self, p):
        self.p = p

    def __call__(self, inputs, train):
        return self.forward(inputs, train)

    def forward(self, inputs, train):
        if train:
            # binomial:生成二项分布随机数的函数
            # 1：保留，0：丢弃）
            # 1 - self.p： 保留概率
            self.mask = np.random.binomial(1, 1 - self.p, size=inputs.shape)
            return inputs * self.mask
        else:
            return inputs
        
    def backward(self, dl_dy):

        # y = mask * x
        # dy/dx = mask
        # dl/dx = dl/dy * dy/dx
        return dl_dy * self.mask
        

if __name__ == "__main__":
    dropout_layer = Dropout(p=0.5)

    # 创建一个输入
    # rand() 均匀分布的浮点数   randn()标准正太分布
    inputs = np.random.rand(5, 5)
    print(inputs)

    train_out = dropout_layer(inputs, train=True)
    print(train_out)

    test_out = dropout_layer(inputs, train=False)
    print(test_out)