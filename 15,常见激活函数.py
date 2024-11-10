import numpy as np

# Relu max(0, x)  输出：0 ~ ∞
class Relu:
    def forward(self, x):
        self.input = x
        self.output = np.maximum(0, x)

        return self.output
    
    def backward(self, d_out):

        return d_out * (self.input > 0)

# sigmoid(x) = 1 / (1 + e^(-x))  输出：0 ~ 1
class Sigmoid:
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))

        return self.output
    
    def backward(self, d_out):
        
        # Sigmoid 的导数 f'(x) = f(x)(1 - f(x))
        return d_out * self.output * (1 - self.output)

# Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)) 输出：-1 ~ 1
class Tanh:
    def forward(self, x):
        self.output = np.tanh(x)

        return self.output
    
    def backward(self, d_out):

         # Tanh 的导数 f'(x) = 1 - f(x)^2
        return d_out * (1 - self.output ** 2)

class Softmax:
    def forward(self, x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        self.output = exp_x / np.sum(exp_x)

if __name__ == "__main__":
    # 定义输入数据
    x = np.array([-1.0, 0.0, 1.0])
    # print(x.shape)
    # 随机生成从上层传下来的梯度
    d_out = np.random.randn(*x.shape)

    # 验证 Relu
    relu = Relu()
    relu_output = relu.forward(x)
    relu_grad = relu.backward(x)
    print("ReLU Forward Output:")
    print(relu_output)
    print("ReLU Backward Output:")
    print(relu_grad)

    # 验证 Sigmoid
    sigmoid = Sigmoid()
    sigmoid_output = sigmoid.forward(x)
    sigmoid_grad = sigmoid.backward(d_out)

    print("\nSigmoid Forward Output:")
    print(sigmoid_output)
    print("Sigmoid Backward Output:")
    print(sigmoid_grad)

    # 验证 Tanh
    tanh = Tanh()
    tanh_output = tanh.forward(x)
    tanh_grad = tanh.backward(d_out)

    print("\nTanh Forward Output:")
    print(tanh_output)
    print("Tanh Backward Output:")
    print(tanh_grad)
         

    