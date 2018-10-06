import numpy as np

A = np.array([
    [3, 2, -1, 4],
    [1, 0, 2, 3],
    [-2, -2, 3, -1]
])

# https://scipy-cookbook.readthedocs.io/items/RankNullspace.html
def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


nullA = nullspace(A)
print("Two linearly independent vectors are:")
print("Vector 1:", nullA[:, [0]])
print("Vector 2:", nullA[:, [1]])


# 2 linearly independent columns
print("A has:", A.shape[1], "columns", "of which", np.linalg.matrix_rank(A)
      , " are linearly independent")
# 2 linearly independent rows
print(np.linalg.matrix_rank(A.T))

# since more columns than rows
# need to get pseudo inverse
A_Inverse = np.linalg.pinv(A)
print("A inverse:", A_Inverse)

