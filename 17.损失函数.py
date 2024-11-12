import numpy as np

# 交叉熵损失（Cross-Entropy Loss），衡量两个概率分布之间的差异
# y_pred：预测 y_true：真实
# 多分类 Loss = -sigma(y_true * log(y_pred))
# 二分类 Loss = -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))

class Cross_Entropy_loss():
    def __init__(self):
        self.y_pred = None
        self.y_true = None
    
    def __call__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return self.forward(self.y_pred, self.y_true)
    
    def forward(self, y_pred, y_true):
        loss = np.sum(-(y_true * np.log(y_pred)), axis=1).mean()
        return loss

if __name__ == "__main__":
    y_true_mult = np.array([[1, 0, 0, 0],    # 类别 0
                            [0, 1, 0, 0],    # 类别 1
                            [0, 0, 1, 0],    # 类别 2
                            [0, 0, 0, 1]])   # 类别 3
 
    y_pred_mult = np.array([[0.7, 0.1, 0.1, 0.1],   # 类别 0 的预测
                            [0.2, 0.6, 0.1, 0.1],   # 类别 1 的预测
                            [0.1, 0.1, 0.7, 0.1],   # 类别 2 的预测
                            [0.1, 0.1, 0.1, 0.7]])  # 类别 3 的预测

    y_true_binary = np.array([1, 0, 1])
    y_pred_binary = np.array([0.9, 0.2, 0.8])
    loss_fn = Cross_Entropy_loss()
    loss = loss_fn(y_pred_mult, y_true_mult)
    print("loss:", loss)