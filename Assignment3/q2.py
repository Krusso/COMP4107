import sklearn.cluster as sk
from sklearn.utils import shuffle
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.75))


input_size = 784
output_size = 10
centroids = 100
X = tf.placeholder("float", shape=[None, input_size])
Y = tf.placeholder("float", shape=[None, output_size])

w_h1 = init_weights([centroids, output_size])

def model(X, w, centroid, b):
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        b = tf.transpose(b)
        #print("stack", sess.run(tf.tile(tf.expand_dims(X, 1), [1, 2, 1]), feed_dict={X: [[0, 0], [1, 1]]}))
        #print("centroid", sess.run(tf.reshape(tf.tile(centroid, [2, 1]), [2, 2, 2]), feed_dict={X: [[0, 0], [1, 1]]}))

        x1 = tf.to_float(tf.tile(tf.expand_dims(X, 1), [1, centroids, 1]))
        centroid1 = tf.to_float(tf.reshape(tf.tile(centroid, [tf.shape(X)[0], 1]), [tf.shape(X)[0],
                                                                                    centroids,
                                                                                    input_size]))

        #print("subtract", sess.run(tf.subtract(x1, centroid1), feed_dict={X: [[0, 0], [1, 1]]}))
        #print("subtract", sess.run(tf.norm(tf.subtract(x1, centroid1), axis=-1), feed_dict={X: [[0, 0], [1, 1]]}))
        dist = tf.square(tf.norm(tf.subtract(x1, centroid1), axis=-1))
        #print("weights inside", sess.run(w))
        #print("dist", sess.run(dist, feed_dict={X: [[0, 0], [1, 1]]}))
        #print("b", b)
        negative = tf.to_float(tf.negative(b))
        #print("negative", sess.run(negative, feed_dict={X: [[0, 0], [1, 1]]}))
        beta = tf.multiply(negative, dist)
        #print("beta", sess.run(beta, feed_dict={X: [[0, 0], [1, 1]]}))
        exponent = tf.exp(beta)
        #print("w", sess.run(w, feed_dict={X: [[0, 0], [1, 1]]}))
        #print("exponent", sess.run(exponent, feed_dict={X: [[0, 0], [1, 1]]}))
        return tf.matmul(exponent, w)


class mnistDataset:
    def __init__(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        trX1, trY1, teX1, teY1 = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

        for i in range(0, len(trX1)):
            image_vector = trX1[i]
            trX1[i] = [j / 255.0 for j in image_vector]

        for i in range(0, len(teX1)):
            image_vector = teX1[i]
            teX1[i] = [j / 255.0 for j in image_vector]

        self.trX = trX1
        self.trY = trY1
        self.teX = teX1
        self.teY = teY1

        self.trX, self.trY = shuffle(self.trX, self.trY)
        self.teX, self.teY = shuffle(self.teX, self.teY)

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

    def getCentroids(self, k=centroids):
        kmeans = sk.KMeans(n_clusters=k, init='random', n_init=1).fit(self.trX)
        print("inertia", kmeans.inertia_)
        return kmeans.cluster_centers_

    def getBetas(self, cluster_centers=None):
        if cluster_centers is None:
            cluster_centers = self.getCentroids()

        sigma = np.zeros([len(cluster_centers)], dtype=np.float32)

        means = cluster_centers
        for x in self.trX:
            best = float("inf")
            index = -1
            for y in range(0, len(means)):
                d = np.linalg.norm(x - means[y])
                if d < best:
                    best = d
                    index = y
            sigma[index] += best

        return tf.divide(1, tf.multiply(2., tf.square(sigma)))


rbf = mnistDataset()
c = rbf.getCentroids()
py_x = model(X, w_h1, centroid=c,
             b=rbf.getBetas(c))


def printAccuracy(data, labels, size, sess):
    accuracy = 0.
    for start, end in zip(range(0, len(data), size), range(size, len(data) + 1, size)):
        accuracy += np.mean(np.argmax(labels[start:end], axis=1) ==
                            sess.run(predict_op, feed_dict={X: data[start:end]}))
    return accuracy


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.AdamOptimizer().minimize(cost)
predict_op = tf.argmax(py_x, 1)

batchSize = 128
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('logs/', graph=sess.graph)
    tf.global_variables_initializer().run()
    summary_writer.add_graph(py_x)
    for i in range(20):
        weightsBefore = sess.run(w_h1)
        for start, end in zip(range(0, len(rbf.trX), batchSize), range(batchSize, len(rbf.trX)+1, batchSize)):
            sess.run(train_op, feed_dict={X: rbf.trX[start:end], Y: rbf.trY[start:end]})

        accuracy = printAccuracy(rbf.trX, rbf.trY, 100, sess)
        print("accuracy # after", accuracy)
        print("accuracy % after", i, accuracy / int((len(rbf.trY) / batchSize)))

        accuracy = printAccuracy(rbf.teX, rbf.teY, 100, sess)
        print("accuracy # after testing", accuracy)
        print("accuracy % after testing", i, accuracy / int((len(rbf.teY) / batchSize)))

        weightsAfter = sess.run(w_h1)
        print("Difference in weights", np.sum(weightsBefore - weightsAfter))

# data = mnistDataset()
# data.kmean()

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
