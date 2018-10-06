# Question 1

import numpy as np
from scipy import spatial

A = [
    [3, 1, 2, 3],
    [4, 3, 4, 3],
    [3, 2, 1, 5],
    [1, 6, 5, 2]
]

uO, sO, vO = np.linalg.svd(A)

k = 2
u = uO[:, :k]
s = np.diag(sO[:k])
v = vO.T[:, :k]

sv = np.dot(s, u[3])
usv = np.dot(v[0], sv)

prediction = 4 + usv
print("prediction is: %.2f" % prediction)

Alice = [5, 3, 4, 4]
# Alice2D
Alice2D = np.dot(np.dot(Alice, v), np.linalg.inv(s))

# finding closest to Alice in 2d
Alicia2D = np.dot(np.dot(A[0], v), np.linalg.inv(s))
Bob2D = np.dot(np.dot(A[1], v), np.linalg.inv(s))
Mary2D = np.dot(np.dot(A[2], v), np.linalg.inv(s))
Sue2D = np.dot(np.dot(A[3], v), np.linalg.inv(s))

print('Alice2d:', Alice2D)
print('Alicia2d:', Alicia2D)
print('Bob2d:', Bob2D)
print('Mary2d:', Mary2D)
print('Sue2d:', Sue2D)

twoD = list([Alicia2D, Bob2D, Mary2D, Sue2D])
names = list(["Alice", "Bob", "Mary", "Sue"])
print(names[spatial.KDTree(twoD).query(Alice2D)[1]], "is closest to Alice in 2d")

# finding closest to Alice in 4d
Alice4D = np.dot(np.dot(Alice, vO.T[:, :4]), np.linalg.inv(np.diag(sO[:4])))
Alicia4D = np.dot(np.dot(A[0], vO.T[:, :4]), np.linalg.inv(np.diag(sO[:4])))
Bob4D = np.dot(np.dot(A[1], vO.T[:, :4]), np.linalg.inv(np.diag(sO[:4])))
Mary4D = np.dot(np.dot(A[2], vO.T[:, :4]), np.linalg.inv(np.diag(sO[:4])))
Sue4D = np.dot(np.dot(A[3], vO.T[:, :4]), np.linalg.inv(np.diag(sO[:4])))

print('Alice2d:', Alice4D)
print('Alicia2d:', Alicia4D)
print('Bob2d:', Bob4D)
print('Mary2d:', Mary4D)
print('Sue2d:', Sue4D)

fourD = list([Alicia4D, Bob4D, Mary4D, Sue4D])
print(names[spatial.KDTree(fourD).query(Alice4D)[1]], "is closest to Alice in 4d")
