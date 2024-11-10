import numpy as np

# 线性层：通过权重矩阵和偏置向量对输入进行线性映射 y = Wx + b
class Linear():
    def __init__(self, input, output):
        scale = np.sqrt(dim_in / 2)
        # np.random.randn(生成标准正态分布的随机数)
        self.weights = np.random.randn(dim_out, dim_in) / scale  # 小值初始化防止梯度爆炸
        self.bias = np.random.randn(dim_out, 1)

    def forward(self, x):
        self.x = x # 缓存输入，用于反向传播计算
        return np.dot(self.weights, self.x) + self.bias
    
    # __call_:使得类的实例 可以直接像函数一样被调用
    def __call__(self, x):
        return self.forward(x)
    
    def backward(self, dl_dy):

        # 计算损失相对于输入、权重和偏置的梯度，假设我们有一个从上层传下来的梯度d_out
        # y = w * x + b
        # dl / dx = dl/dy * dy/dx
        # dy/dy =  w.T

        dl_dx = np.dot(self.weights.T, dl_dy) # (input_dim, output_dim) * (output_dim, batch_size) = (input_dim, batch_size)
        # dl / dw = dl/dy * dy/dw;  dy/dw = x.T
        dl_dw = np.dot(dl_dy, self.x.T)   # (output_dim, batch_size) * (batch_size, input_dim) = (output_dim, input_dim)
        
        dl_db = np.sum(dl_dy, axis=1)     # 每个输出单元的所有样本贡献的梯度相加, 在样本维度（axis 1）上对dl/dy求和

        return dl_dx, [dl_dw, dl_db]

if __name__ == "__main__":
    # 定义输入和输出维度
    dim_in = 3
    dim_out = 2

    # 创建一个Linear层的实例
    linear_layer = Linear(dim_in, dim_out)

    # 生成一个形状为（3， 5）的输入，表示有5个样本，每个样本有三个特征
    x = np.random.randn(dim_in, 5) 
    output = linear_layer(x)  # 实际上是调用了 linear_layer.__call__(x),等价于linear_layer.forward(x)

    print(f"x:\n{x}")
    print(f"output:\n{output}")

    dl_dy = np.random.randn(dim_out, 5)
    dl_dx, grads = linear_layer.backward(dl_dy)
    dl_dw, dl_db = grads

    print("backford")
    print(f"dl_dx:\n{dl_dx}")
    print(f"dl_dw:\n{dl_dw}")
    print(f"dl_db:\n{dl_db}")
