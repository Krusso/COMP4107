import numpy as np

A = np.array([
    [3, 2, -1, 4],
    [1, 0, 2, 3],
    [-2, -2, 3, -1]
])

# 2 linearlly indepdent columns
print(np.linalg.matrix_rank(A))
# 2 linearlly indepdent rows
print(np.linalg.matrix_rank(A.T))

# since more columns than rows
# need to get pseudo inverse
A_Inverse = np.linalg.pinv(A)
print(A_Inverse)
print(np.dot(A, A_Inverse))

