import argparse
import sys
import sklearn.cluster as sk
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import math


# creates activation function
def gaussian_function(input_layer):
    initial = math.exp(-2*math.pow(input_layer, 2))
    return initial

np_gaussian_function = np.vectorize(gaussian_function)

def d_gaussian_function(input_layer):
    initial = -4 * input_layer * math.exp(-2*math.pow(input_layer, 2))
    return initial

np_d_gaussian_function = np.vectorize(d_gaussian_function)

np_d_gaussian_function_32 = lambda input_layer: np_d_gaussian_function(input_layer).astype(np.float32)

def tf_d_gaussian_function(input_layer, name=None):
    with ops.name_scope(name, "d_gaussian_function", [input_layer]) as name:
        y = tf.py_func(np_d_gaussian_function_32, [input_layer],[tf.float32], name=name, stateful=False)
    return y[0]

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    rnd_name = 'PyFunGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def gaussian_function_grad(op, grad):
    input_variable = op.inputs[0]
    n_gr = tf_d_gaussian_function(input_variable)
    return grad * n_gr

np_gaussian_function_32 = lambda input_layer: np_gaussian_function(input_layer).astype(np.float32)

def tf_gaussian_function(input_layer, name=None):
    with ops.name_scope(name, "gaussian_function", [input_layer]) as name:
        y = py_func(np_gaussian_function_32, [input_layer], [tf.float32], name=name, grad=gaussian_function_grad)
    return y[0]
# end of defining activation function



class mnistDataset:
    def __init__(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        trX1, trY1, teX1, teY1 = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

        self.trX = []
        self.trY = []
        for i in range(len(trX1)):
            if np.array_equal(trY1[i], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]) or \
                    np.array_equal(trY1[i], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]):

                self.trX.append(trX1[i])
                self.trY.append(trY1[i])

    def kmean(self):
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter('logs/', graph=sess.graph)

            for k in range(2, 50, 5):
                avg = 0
                for _ in range(0, 10, 1):
                    kmeans = sk.KMeans(n_clusters=k, init='random', n_init=1).fit(self.trX)
                    avg += kmeans.inertia_
                    print(k, kmeans.inertia_)

                kmeans = sk.KMeans(n_clusters=k, n_init=1).fit(self.trX)
                print(k, "using best fit", kmeans.inertia_)
                summary_writer.add_summary(tf.Summary(value=[
                    tf.Summary.Value(tag="objective function", simple_value=kmeans.inertia_/10),
                ]), k)


data = mnistDataset()
data.kmean()

# keep for now :)
# kmeansDistance = kmeans.transform(self.trX)
#
# variance = 0
# i = 0
# for label in kmeans.labels_:
#     variance = variance + kmeansDistance[i][label]
#     i = i + 1
#
# print(variance)
#
# means = kmeans.cluster_centers_
# s = 0
# for x in self.trX:
#     best = float("inf")
#     for y in means:
#         d = np.linalg.norm(x - y)
#         if d < best:
#             best = d
#     s += best
# print(s, "*****")
