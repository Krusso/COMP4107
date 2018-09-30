# Question 1

import numpy as np 

A = [
    [3, 1, 2, 3],
    [4, 3, 4, 3],
    [3, 2, 1, 5],
    [1, 6, 5, 2]
]

u, s, v = np.linalg.svd(A)
print(u)
u = [i[:2] for i in u]
print(u)
print(s)
s = np.multiply(s, np.identity(4))
print(s)
s = [i[:2] for i in s]
v = [i[:2] for i in v.T]
print(v)

sv = np.dot(s[:2], u[3])
usv = np.dot(v[0], sv)

prediction = 4 + usv
print("prediction is: %.2f" % prediction)