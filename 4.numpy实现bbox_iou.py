import numpy as np

def IOU(boxA, boxB):
    #计算交集
    xA = max(boxA[0], boxB[0])          # 左上角
    yA = max(boxA[1], boxB[1])          # 左上角
    xB = min(boxA[2], boxB[2])          # 右下角
    yB = min(boxA[3], boxB[3])          # 右下角

    # 计算交集面积
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # 计算两个框的面积
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # 计算交并比
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

def IOU_numpy(boxsA, boxsB):
    #计算所有框的面积
    #多维数组的切片，array[行, 列]
    #批量处理boxsA(n, 4)   boxsB(m, 4)
    boxsAArea = (boxsA[:, 2] - boxsA[:, 0]) * (boxsA[:, 3] - boxsA[:, 1])
    boxsBArea = (boxsB[:, 2] - boxsB[:, 0]) * (boxsB[:, 3] - boxsB[:, 1])

    #左上角，右下角
    #A:(n, 4) -> (n, 1, 2) n个框，每个框左上角坐标数量1个，坐标的维度2维
    #B:(m, 2) 
    #广播(n, m, 2)
    top_left = np.maximum(boxsA[:, None, :2], boxsB[:, :2])
    bottom_right = np.minimum(boxsA[:, None, 2:], boxsB[:, 2:])

    #wh(n, m, 2) n:boxsA中框的数量，m:boxsB中框的数量，2:坐标(x1, y1)
    wh = np.clip(bottom_right - top_left, a_min=0,a_max=None)

    #interArea: (n, m)
    interArea = wh[:, :, 0] * wh[:, :, 1]

    union = boxsAArea[:, None] + boxsBArea - interArea

    iou = interArea / union

    return iou


if __name__ == "__main__":
    boxsA = np.array([[1, 1, 3, 3],  # 框1
                      [2, 2, 4, 4],  # 框2
                      [0, 0, 2, 2]]) # 框3

    boxsB = np.array([[1, 1, 2, 2],  # 框A1
                      [0, 0, 1, 1]]) # 框A2

    # 计算IoU
    iou_result = IOU_numpy(boxsA, boxsB) 
    print(iou_result)



