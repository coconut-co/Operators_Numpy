import numpy as np

# 使用 SGD（随机梯度下降） 优化线性回归的参数 y = w * x + b，找到最优的 w 和b让预测值 y' 和真实值 y 尽可能接近
# 使用均方误差损失函数来衡量 y' 和 y 的差距
class SGD():
    def __init__(self, learning_rate, momentum):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}  # 保存参数的动量

    def updata(self, params, grads):
        # 初始化动量为 0 
        for key in params.keys():
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            
            # 更新动量： v = momentum * v - learning_rate * grad
            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * grads[key]
            
            # 更新参数： param = param + v
            params[key] = np.reshape(params[key], self.velocity[key].shape)  # 调整形状
            params[key] += self.velocity[key]

        return params

if __name__ == "__main__":

    np.random.seed(42)
    x = np.random.randn(10, 1)  # 10个样本，一个特征
    y = 3 * x + 2 + np.random.randn(10, 1) * 0.1  # 添加一些噪声

    # 初始化参数
    params = {
        "w": np.random.randn(1),  # 权重
        "b": np.random.randn(1)   # 偏置
    }
    learning_rate = 0.01
    momentum = 0.9
    epochs = 100

    # 创建 SGD 优化器
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum)

    # 训练过程
    for epoch in range(epochs):
        # 打乱数据
        indices = np.random.permutation(len(x))
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        for i in range(len(x)):
            x_i = x_shuffled[i: i+1]
            y_i = y_shuffled[i: i+1]

            # 预测
            y_pred = params["w"] * x_i + params["b"]

            # 计算梯度
            grad_w = 2 * (y_pred - y_i) * x_i
            grad_b = 2 * (y_pred - y_i).squeeze()

            # 使用 SGD 更新参数
            grads = {"w": grad_w, "b": grad_b}
            params = optimizer.updata(params, grads)

        if (epoch + 1) % 10 == 0:
            y_pred_all = params["w"] * x + params["b"]
            loss = np.mean((y_pred_all - y) ** 2)
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    print(f"Trained parameters: w = {params['w'][0].item():.4f}, b = {params['b'][0].item():.4f}")

