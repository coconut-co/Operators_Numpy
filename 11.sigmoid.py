import numpy as np

# sigmoid = 1 / (1 + exp(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sofrmax = exp(x) / sum(exp(x))
def softmax(x):
    # 防止分布极端，减去最大值
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


if __name__ == "__main__":
    x = np.array([1, 0.9, 0.7])
    print(sigmoid(x))
    print(softmax(x))