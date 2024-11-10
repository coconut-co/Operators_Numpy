# Diceloss 图像分割类损失，关注与目标区域的重叠部分，减少小区域的忽略现象
# 交叉熵损失（cross-entropy-loss）图像分类任务损失，最小化预测概率与真实标签概率的差异

import numpy as np

def dice_loss(y_true, y_pred, eps=1e-6):

    # 将预测值和真实标签展平为一维数组,
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    inter = np.sum(y_true_flat * y_pred_flat)
    union = np.sum(y_true_flat) + np.sum(y_pred_flat)

    # eps防止0/0的情况，预测和真实标签完全为零的情况（如某个类完全没有预测到，也不在标签中），导致分子和分母都为 0。
    dice_score = (2 * inter + eps) / (union + eps)

    return 1-dice_score

# 定义测试函数
def test_diceloss():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.9, 0.8, 0.2, 0.1])

    loss = dice_loss(y_true, y_pred)

    # 手动计算
    eps = 1e-5
    intersection = 1.7  # (1*0.9 + 1*0.8 + 0*0.2 + 0*0.1)
    union = 4.0  # (1+0.9 + 1+0.8 + 0+0.2 + 0+0.1)
    expected_loss = 1 - (2 * intersection + eps) / (union + eps)

    print(f"Diceloss:      {loss}")
    print(f"expected_loss: {expected_loss}")

if __name__ == "__main__":
    test_diceloss()

