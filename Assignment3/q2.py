import argparse
import sys
import sklearn.cluster as sk
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


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

            for k in range(2, 100, 1):
                kmeans = sk.KMeans(n_clusters=k).fit(self.trX)

                print(kmeans.inertia_)
                summary_writer.add_summary(tf.Summary(value=[
                    tf.Summary.Value(tag="objective function", simple_value=kmeans.inertia_),
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
