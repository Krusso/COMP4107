import numpy as np

table = [["Step Size", "Iterations", "x"]]
tolerance = 0.01

A = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [4, 5, 6],
    [1, 1, 1]
])

b = np.array([1, 1, 1, 1])


def t(x):
    try:
        return np.dot(A.T, np.dot(A, x)) - np.dot(A.T, b)
    except RuntimeWarning:
        print("Run time warning: ", x)
        return x


ix = x = np.random.rand(3)

for stepsize in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5]:
    x = ix
    iterations = 0
    while np.linalg.norm(t(x), 2) > tolerance:
        try:
            print(x)
            print(t(x))
            x = x - np.dot(stepsize, t(x))
        except RuntimeWarning:
            break
        iterations += 1
    table.append([stepsize, iterations, x])

print(table)
