import argparse
import sys
import sklearn.cluster as sk
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


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
        kmeans = sk.KMeans(n_clusters=2).fit(self.trX)

        print(kmeans.inertia_)

        kmeansDistance = kmeans.transform(self.trX)

        variance = 0
        i = 0
        for label in kmeans.labels_:
            variance = variance + kmeansDistance[i][label]
            i = i + 1

        print(variance)

        means = kmeans.cluster_centers_
        s = 0
        for x in self.trX:
            best = float("inf")
            for y in means:
                d = (x - y).T.dot(x - y)
                if d < best:
                    best = d
            s += best
        print(s, "*****")


data = mnistDataset()
data.kmean()


print("\n")
kmeans = sk.KMeans(n_clusters=1).fit([[0, 0], [1, 1]])
kmeansDistance = kmeans.transform([[0, 0], [1, 1]])

np.set_printoptions(threshold=np.nan)
print(kmeans.cluster_centers_)
print(kmeans.inertia_)
print(kmeans.score([[0, 0], [1, 1]]))
print(type(kmeans.inertia_))

variance = 0
i = 0
for label in kmeans.labels_:
    variance = variance + kmeansDistance[i][label]
    i = i + 1

print(variance)
