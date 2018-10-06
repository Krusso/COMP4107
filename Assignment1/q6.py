# https://sikaman.dyndns.org:8443/WebSite/rest/site/courses/4107/documents/Lassiter_2012_2013.pdf

# mnist has its own file format
# https://gist.github.com/akesling/5358964


import os
import struct
import numpy as np

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""


def read(dataset="training", path="./"):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    print(os.listdir(path))

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)


def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

def draw(column):
    column.resize(28, 28)
    print(column)
    show(column)

def transform(matrix):
    return np.ravel(matrix)


def getByLabel(label, mnist):
    try:
        while True:
            number = next(mnist)
            if number[0] == label:
                return number
    except StopIteration:
        return

A = {
    0: np.zeros((28 * 28, 0)),
    1: np.zeros((28 * 28, 0)),
    2: np.zeros((28 * 28, 0)),
    3: np.zeros((28 * 28, 0)),
    4: np.zeros((28 * 28, 0)),
    5: np.zeros((28 * 28, 0)),
    6: np.zeros((28 * 28, 0)),
    7: np.zeros((28 * 28, 0)),
    8: np.zeros((28 * 28, 0)),
    9: np.zeros((28 * 28, 0)),
}


mnist = read("training")
mnistTesting = read("testing")
i = 0
while True:
    try:
        i = i+1
        number = next(mnist)
        label = number[0]
        column = transform(number[1])
        A[label] = np.c_[A[label], column]
        if i == 5000:
            break
    except StopIteration:
        break


# printing entire arrays
np.set_printoptions(threshold=np.nan)


image = getByLabel(3, mnist)
unknown = transform(image[1])
array = []
tmp = np.identity(784)
# for y in range(len(A)):
#     u, s, v = np.linalg.svd(A[y])
#     residual = np.linalg.norm(np.dot((tmp - np.dot(u[:,:10], u[:,:10].T)), unknown), 2)
#     print(y, " Residual ", residual)
#     array.append(residual)


testCases = []
i = 0
while True:
    try:
        number = next(mnistTesting)
        label = number[0]
        column = transform(number[1])
        if label != 4:
            continue
        testCases.append((label, column))
        if i == 100:
            break
        i = i+1
    except StopIteration:
        break

basis = []
for b in [1, 2, 5, 6] + list(range(10, 50, 3)):
    basis.append((b, []))
    print("Calculating svd for basis sized: ", b)
    for y in range(len(A)):
        u, s, v = np.linalg.svd(A[y])
        basis[len(basis) - 1][1].append(np.dot(u[:,:b], u[:,:b].T))


for b in basis:
    correct = 0
    totalTc = 0
    for tc in testCases:
        array = []
        i = 0
        for y in b[1]:
            array.append((i, np.linalg.norm(np.dot(tmp - y, tc[1]), 2)))
            i = i + 1
        array.sort(key=lambda x: x[1])
        #print("tc done")
        if array[0][0] == tc[0]:
            correct = correct + 1
        totalTc = totalTc + 1
    print("For b: ", b[0], " Correct: ", correct, " Total test cases: ", totalTc, " Percentage: ", (correct/totalTc))