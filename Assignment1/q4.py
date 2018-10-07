import numpy as np


A = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [4, 5, 6],
    [1, 1, 1]
])

b = np.array([1, 1, 1, 1]).T

x, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)

print(x)
