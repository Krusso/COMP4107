# Question 2, Compute the SVD for the matrix
# [ 1 2 3 ]
# [ 2 3 4 ]
# [ 4 5 6 ]
# [ 1 1 1 ]

import numpy as np 
A = [
    [1, 2, 3],
    [2, 3, 4],
    [4, 5, 6],
    [1, 1, 1]
]
u, s, v = np.linalg.svd(A)

