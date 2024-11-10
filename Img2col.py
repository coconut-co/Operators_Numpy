import numpy as np

class Img2colIndices():
    def __init__(self, kernel_size, padding, stride):
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
    
    # 将feature map 按照卷积核的大小转换为适合矩阵乘法的形式
    def get_img2col_indices(self, out_h, out_w):
        
        # 位置编码
        # eg：kernel_size: 2 × 2，feature_map: 4 × 4
        