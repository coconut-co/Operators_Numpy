import numpy as np

# Cos similarity = (A·B) / (||A|| * ||B||)
def cos_similarity(vector1, vector2):
    # 计算点积
    dot_product = np.dot(vector1, vector2)
    # np.linalg.norm: 计算向量或矩阵的范数，默认计算欧几里得范数（sqrt(x1**2 + x2**)）,即模长
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    similarity = dot_product / (norm1 * norm2)
    return similarity

if __name__ == '__main__':
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([2, 3, 4])
    print(cos_similarity(vector1, vector2))   # 0.9925833339709303