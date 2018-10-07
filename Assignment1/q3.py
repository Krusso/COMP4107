# Question 3
import numpy as np 
import math

# Specify givens:
l = lambda i: -0.7 + 0.001*(i-1)  # equation for xi = yi = -0.7 + 0.001(i-1)
r = range(1, 1402)                # this is our range (we start at 0 due to indexing so it really goes from 0 to 1400)
k = 2                             # we are computing the best rank(2) for matrix A

# Generate the matrix A based on a_ij = sqrt(1 - x_i ^2 - y_i ^2)
A = [[math.sqrt(1-l(i)**2 - l(j)**2) for j in r] for i in r]
u, s, v = np.linalg.svd(A)
vt = v[:k] # we get the first rows of v
uk = np.array([u[:, i] for i in range(0, k)]).T
sk = np.multiply(s[:k], np.identity(k)) # first k singular values

Ak = np.dot(uk, np.dot(sk, vt))
# We have computed A_(k=2), now determine |A - A_2|
print("Norm", np.linalg.norm(A-Ak), )
