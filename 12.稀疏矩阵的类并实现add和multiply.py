import numpy as np

class sparseMatrix:
    def __init__(self, matrix):
        # 将稀疏矩阵转化为坐标coo(坐标)格式:[(row, col, value)]
        self.matrix = []
        self.rows = len(matrix)
        self.cols = len(matrix[0])
        for i in range(self.rows):
            for j in range(self.cols):
                if matrix[i][j] != 0:
                    self.matrix.append((i, j, matrix[i][j]))
    
    def add(self, other_matrix):
        # {
        #     (0, 0): 1,
        #     (1, 1): 2,
        #     (2, 0): 3
        # }
        result = {(i, j):value for i, j, value in self.matrix}
        # 遍历另一个矩阵的非0元素并相加
        for i, j, value in other_matrix.matrix:
            if (i, j) in result:
                result[(i, j)] += value
            else:
                result[(i, j)] = value
        
        # 将结果转换为list
        max_row = max(i for i, j, _ in self.matrix + other_matrix.matrix) + 1 # 计算结果矩阵的行数
        max_col = max(j for i, j, _ in self.matrix + other_matrix.matrix) + 1 # 计算结果矩阵的列数

        return self.dict2list(max_row, max_col, result)
    
    def dict2list(self, max_row, max_col, result):
        result_matrix = [[0] * max_col for _ in range(max_row)]

        for (i, j), value in result.items():
            result_matrix[i][j] = value
        return sparseMatrix(result_matrix)        
    
    def __str__(self):
        # 打印稀疏矩阵
        max_row = max(i for i, j, _ in self.matrix) + 1
        max_col = max(j for i, j, _ in self.matrix) + 1
        matrix = np.zeros((max_row, max_col), dtype=int)
        for i, j, value in self.matrix:
            matrix[i][j] = value
        return "\n".join(" ".join(map(str, row)) for row in matrix)
    
if __name__ == "__main__":
    matrix1 = np.array([[1, 0, 0], [0, 2, 0], [3, 0, 0]])
    matrix2 = np.array([[0, 4, 1], [5, 0, 0], [0, 6, 0]])

    sparematrix1 = sparseMatrix(matrix1)
    sparematrix2 = sparseMatrix(matrix2)

    matrix_add = sparematrix1.add(sparematrix2)


    print(f"matrix1:\n{sparematrix1}")
    print(f"matrix2:\n{sparematrix2}")
    print(f"add:\n{matrix_add}")
