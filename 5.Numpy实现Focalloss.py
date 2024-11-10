# 针对类别不平衡问题，标准交叉熵损失正负样本的权重一样
# 因此在负样本较多的情况下，损失函数往往受大量负样本主导
# Focal Loss 通过给容易分类的样本降低权重，突出难分类样本的贡献

import numpy as np

def multiclass_focal_loss_fun(y_true, y_pred, alpha = 0.5, gamma = 2, class_weights=None):
    # 三元运算符 where(condition ? true_value : false_value)
    p_t = np.where(y_true == 1, y_pred, 1 - y_pred)   # 样本难以分类成度
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha) # 正负样本的权重

    # 计算公式
    focal_loss = -alpha_t * (1 - p_t) ** gamma * np.log(p_t)

    if class_weights is None:
        # 如果没有类别权重，计算平均损失
        focal_loss = np.mean(focal_loss)
    else:
        # 如果有类别权重，按权重调整损失
        # multiply:逐元素相乘
        focal_loss = np.sum(np.multiply(focal_loss, class_weights))

    return focal_loss

y_true = np.array([1, 0, 1, 0])
y_pred = np.array([0.8, 0.4, 0.7, 0.2])

class_weights = np.array([0.7, 0.3, 0.7, 0.3])

loss = multiclass_focal_loss_fun(y_true, y_pred, class_weights=class_weights)
print(loss)
