import numpy as np

tolerance = 0.01
stepsize = 0.01  # E

A = np.array([
    [1, 2, 4, 1],
    [2, 3, 5, 1],
    [3, 4, 6, 1]
]).T

b = np.array([1, 1, 1, 1])

# Some complaining TODO: michael look at if this error is cause of something we should fix
# C:/Users/kwojc/PycharmProjects/COMP4107/Assignment1/q4.py:15: FutureWarning: `rcond` parameter will change to the
# default of machine precision times ``max(M, N)`` where M and N are the input matrix dimensions.
# To use the future default and silence this warning we advise to pass `rcond=None`, to keep using the old, explicitly
# pass `rcond=-1`.
#   x, res, rank, sv = np.linalg.lstsq(A,b)
x, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)

print(x)
