# Question 1

import numpy as np 

A = [
    [3, 1, 2, 3],
    [4, 3, 4, 3],
    [3, 2, 1, 5],
    [1, 6, 5, 2]
]

u, s, v = np.linalg.svd(A)
print('shapes')
print(u.shape, s.shape, v.shape)
print('usv',np.dot(u*s,v))

k=2
print('u:',u)
u = [i[:k] for i in u]
print(u)
print(s)
s = np.diag(s[:k])
print('s',s)
v = [i[:k] for i in v.T]
print('v',v)

sv = np.dot(s, u[3])
usv = np.dot(v[0], sv)

prediction = 4 + usv
print("prediction is: %.2f" % prediction)

Alice = [5,3,4,4]
Alicia = [3,1,2,3]
Bob = [4,3,4,3]
## Alice2D
Alice2D = np.dot(np.dot(Alice, v),np.linalg.inv(s))
Alice2D=[0.64, -0.30]

#using Alice2D to predict others
sv = np.dot(s, u[0])
usv = np.dot(Alice2D, sv)
prediction = 4 + usv
print("prediction is: %.2f" % prediction)

Alicia2D = np.dot(np.dot(Alicia, v),np.linalg.inv(s))
Bob2D = np.dot(np.dot(Bob, v),np.linalg.inv(s))

print('Alice2d:', Alice2D)
print('Alicia2d:', Alicia2D)
print('Bob2d:', Bob2D)
####find alice 4d
# u, s, v = np.linalg.svd(A)
# v = v.T
# print(v)
# Alice4d = np.dot(np.dot(Alice, v), np.linalg.inv(s)
# print(Alice4d)