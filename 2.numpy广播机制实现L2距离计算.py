import numpy as np

#cloud1到cloud2的L2距离
def L2_1(cloud1,cloud2):
    m, n = len(cloud1), len(cloud2)
    print(cloud1.shape)         #(m,d)(3,2)
    print(cloud2.shape)         #(n,d)(2,2)

    # cloud1: (m, d) -> (n, m, d) -> (m, n, d)
    # cloud2: (n, d) -> 广播 -> (m, n, d) 
    cloud1 = np.repeat(cloud1, n, axis=0)       # (m, d) -> (m*n, d)
    cloud1 = np.reshape(cloud1, (m, n, -1))     # (m*n, d) -> (m, n, d)

    dist = np.sqrt(np.sum((cloud1 - cloud2)**2, axis=2))
    return dist


def L2_2(cloud1, cloud2):

    cloud2 = cloud2[:, None, :]                 # (n, d) -> (n, 1, d)

    dist = np.sqrt(np.sum((cloud1 - cloud2)**2, axis=2))
    return dist

def L2_3(cloud1, cloud2):

    cloud1 = np.expand_dims(cloud1, axis=1)     # (m, d) -> (m, 1, d)

    dist = np.sqrt(np.sum((cloud1 - cloud2)**2, axis=2))
    return dist

if __name__ == "__main__":
    cloud1 = np.array([[1, 2], [3, 4], [5, 6]])
    cloud2 = np.array([[7, 8], [9, 10]])

    # 计算成对的L2距离
    distance_matrix_1 = L2_1(cloud1, cloud2)
    distance_matrix_2 = L2_2(cloud1, cloud2)
    distance_matrix_3 = L2_3(cloud1, cloud2)
    print("The pairwise L2 distance matrix is:")
    print(distance_matrix_1)
    print(distance_matrix_2)
    print(distance_matrix_3)