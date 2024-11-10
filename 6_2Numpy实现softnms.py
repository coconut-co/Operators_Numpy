# nms会将与置信度得分最高的框 有较大重叠的 其他框直接移除
# softnms不会直接移除重叠度较高的边界框，而是根据重叠程度降低其他框的置信度，保留一些重叠较多但仍然有用的框

import numpy as np

# A score:0.8
# B score:0.9
# c score:0.9

# Iou(A, B) = 0.99

# 特点：没有硬性指定的Iou阈值

# score_threshold丢弃置信度得分过低的框
def softnms_fun(boxes, scores, sigma=0.5, score_threshold=0.001):

    # 坐标和面积
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (y2 - y1) * (x2 - x1)

    # 选出框A 与其余所有框比较
    for i in range(len(scores)):
        max_idx = i
        max_score = scores[i]

        # 与其他边界框计算iou
        for j in range(i + 1, len(scores)):
            if scores[j] > score_threshold:
                x11 = np.maximum(x1[i], x1[j])
                y11 = np.maximum(y1[i], y1[j])
                x22 = np.minimum(x2[i], x2[j])
                y22 = np.minimum(y2[i], y2[j])

                w = np.maximum(0, x22 - x11)
                h = np.maximum(0, y22 - y11)
                inter_areas = w * h

                iou = inter_areas / (areas[i] + areas[j] - inter_areas)
                weights = np.exp(-(iou * iou) / sigma)
                scores[j] = scores[j] * weights

                # 保留置信度最高的边界框
                if scores[j] > max_score:
                    max_idx = j
                    max_score = scores[j]

        # 交换置信度最高的边界框和当前边界框的位置
        bboxes[i], bboxes[max_idx] = bboxes[max_idx], bboxes[i]
        scores[i], scores[max_idx] = scores[max_idx], scores[i]

    # 过滤置信度低于阈值的边界框
    selected_idx = np.where(scores > score_threshold)
    bboxes = bboxes[selected_idx]
    scores = scores[selected_idx]

    return bboxes, scores


# 输入边界框的坐标
boxes = np.array([[181, 181, 500, 500],
                  [105, 105, 230, 230],
                  [180, 180, 500, 500]])
# 输入每个边界框的置信分数
scores = np.array([0.9, 0.75, 0.95])