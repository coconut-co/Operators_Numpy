# 1.对所有的框进行排序，选择置信度最高的框
# 2.计算剩余框与该框的重叠度（iou交并比）
# 3.设置一个阈值，如果大于阈值则不要，因为这两个框标定同一个物体
# 4.重复，将所有框都遍历一边

import numpy as np

def nms_fun(boxes, scores, iou_threshold):
    # 计算每个边界框的面积
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # 按置信度从高到低排序
    order = scores.argsort()[::-1]          # [0, 2, 1]
    # 保留框的索引
    result = []        

    while order.size > 0:
        i = order[0]    # 当前置信度最高的框的索引
        result.append(i)

        # 计算当前框与其他框的iou
        x11 = np.maximum(x1[i], x1[order[1:]])
        y11 = np.maximum(y1[i], y1[order[1:]])
        x22 = np.minimum(x2[i], x2[order[1:]])
        y22 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, x22 - x11)
        h = np.maximum(0, y22 - y11)
        inter_area = w * h

        # 计算iou
        iou = inter_area / (areas[i] + areas[order[1:]] - inter_area)
        
        # 找出iou小于阈值的索引，保留这些框
        inds = np.where(iou <= iou_threshold)[0]

        # 更新order，保留符合条件的框
        order = order[inds + 1]

    return result

# 输入边界框的坐标
boxes = np.array([[181, 181, 500, 500],
                  [105, 105, 230, 230],
                  [180, 180, 500, 500]])
# 输入每个边界框的置信分数
scores = np.array([0.9, 0.75, 0.95])
# iou的阈值
iou_threshold = 0.5

# 获得保留框的索引
result_indices = nms_fun(boxes, scores, iou_threshold)

print("保留框的索引：", result_indices)
print("保留的框：", boxes[result_indices])
