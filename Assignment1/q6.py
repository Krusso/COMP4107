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
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()


def transform(matrix):
    return np.ravel(matrix)


A = {
    0: np.zeros((28 * 28, 1)),
    1: np.zeros((28 * 28, 1)),
    2: np.zeros((28 * 28, 1)),
    3: np.zeros((28 * 28, 1)),
    4: np.zeros((28 * 28, 1)),
    5: np.zeros((28 * 28, 1)),
    6: np.zeros((28 * 28, 1)),
    7: np.zeros((28 * 28, 1)),
    8: np.zeros((28 * 28, 1)),
    9: np.zeros((28 * 28, 1)),
}

mnist = read()
i = 0
while True:
    try:
        i = i+1
        number = next(mnist)
        label = number[0]
        column = transform(number[1])
        # print(column)
        # print(A[label])
        A[label] = np.c_[A[label], column]
        if i % 100 == 0:
            print(i)
            print(column)
        if i == 10:
            break
    except StopIteration:
        print("done")
        print(i)
        break

print(A[0][1])
u, s, v = np.linalg.svd(A[0], )
# five = next(mnist)
# print(len(five[1]))
# print(len(five[1][0]))
# print(transform(five[1]))
# print(next(mnist))
