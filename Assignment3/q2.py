import sklearn.cluster as sk
from sklearn.utils import shuffle
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from sklearn.model_selection import KFold
import os

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.75))


input_size = 784
output_size = 10
centroids = 90
X = tf.placeholder("float", shape=[None, input_size])
Y = tf.placeholder("float", shape=[None, output_size])

w_h1 = init_weights([centroids, output_size])

def model(X, w, centroid, b, keep_prob=0.5):
    #with tf.Session() as sess:
    #tf.global_variables_initializer().run()
    b = tf.transpose(b)
    #print("stack", sess.run(tf.tile(tf.expand_dims(X, 1), [1, 2, 1]), feed_dict={X: [[0, 0], [1, 1]]}))
    #print("centroid", sess.run(tf.reshape(tf.tile(centroid, [2, 1]), [2, 2, 2]), feed_dict={X: [[0, 0], [1, 1]]}))

    x1 = tf.to_float(tf.tile(tf.expand_dims(X, 1), [1, centroids, 1]))

    # dropout = tf.nn.dropout(w, keep_prob=keep_prob)

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
    dropout = tf.nn.dropout(w, keep_prob=keep_prob)
    return tf.matmul(exponent, dropout)


class mnistDataset:
    def __init__(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        trX1, trY1, teX1, teY1 = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

        self.data = np.append(trX1, teX1, axis=0)
        self.labels = np.append(trY1, teY1, axis=0)

        self.data, self.labels = shuffle(self.data, self.labels)

    def kmean(self):
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter('logs/', graph=sess.graph)

            for k in range(2, 50, 5):
                avg = 0
                for _ in range(0, 10, 1):
                    kmeans = sk.KMeans(n_clusters=k, init='random', n_init=1).fit(self.data)
                    avg += kmeans.inertia_
                    print(k, kmeans.inertia_)

                kmeans = sk.KMeans(n_clusters=k, n_init=1).fit(self.data)
                print(k, "using best fit", kmeans.inertia_)
                summary_writer.add_summary(tf.Summary(value=[
                    tf.Summary.Value(tag="objective function", simple_value=avg/10),
                ]), k)

    def getCentroids(self, k=centroids):
        kmeans = sk.KMeans(n_clusters=k, init='random', n_init=1).fit(self.data)
        print("inertia", kmeans.inertia_)
        return kmeans.cluster_centers_

    def getBetas(self, cluster_centers=None):
        if cluster_centers is None:
            cluster_centers = self.getCentroids()

        sigma = np.zeros([len(cluster_centers)], dtype=np.float32)
        size = np.zeros([len(cluster_centers)], dtype=np.float32)

        means = cluster_centers
        for x in self.data:
            best = float("inf")
            index = -1
            for y in range(0, len(means)):
                d = np.linalg.norm(x - means[y])
                if d < best:
                    best = d
                    index = y
            sigma[index] += best
            size[index] += 1

        sigma = np.array([sigma[j] / size[j] for j in range(len(sigma))])

        print(sigma)
        return tf.divide(1, tf.multiply(2., tf.square(sigma)))


def printAccuracy(data, labels, size, sess):
    accuracy = 0.
    for start, end in zip(range(0, len(data), size), range(size, len(data) + 1, size)):
        accuracy += np.mean(np.argmax(labels[start:end], axis=1) ==
                            sess.run(predict_op, feed_dict={X: data[start:end]}))
    return accuracy


attempt = 0

while True:
    if os.path.isdir('./logs/attempt_{}'.format(attempt)):
        attempt+=1
    else:
        break

k_fold_accuracy = 0
k_folds = 10
fold = 0
for keep_prob in [0.5, 1.0]:
    #try with dropout and without
    rbf = mnistDataset()
    c = rbf.getCentroids()
    py_x = model(X, w_h1, centroid=c,
                 b=rbf.getBetas(c), keep_prob=keep_prob)
    predict_op = tf.argmax(py_x, 1)
    y_true_cls = tf.argmax(Y, dimension=1)
    batch_accuracies = tf.placeholder("float", [None])
    num_batches = tf.placeholder("float")
    with tf.name_scope('accuracy'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
        correct_prediction = tf.equal(predict_op, y_true_cls)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        mean_accuracy = tf.reduce_mean(tf.cast(batch_accuracies, tf.float32))

    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    # train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

    #summaries
    tf.summary.scalar('accuracy', mean_accuracy)

    batchSize = 50
    testSize = 100

    for train_index, test_index in KFold(n_splits=k_folds).split(rbf.data):
        fold += 1
        print(train_index, test_index)
        trX, teX = rbf.data[train_index], rbf.data[test_index]
        trY, teY = rbf.labels[train_index], rbf.labels[test_index]
        print(len(trX), len(trY))
        result_dir = './logs/attempt_{}/fold_{}_keep_prob_{}/'.format(attempt,fold, int(keep_prob*100))
        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(result_dir, graph=sess.graph)
            tf.global_variables_initializer().run()
            accuracy = printAccuracy(trX, trY, 100, sess)
            print("accuracy # before running", accuracy)
            print("accuracy % before running", accuracy / int((len(trY) / batchSize)))

            accuracy = printAccuracy(teX, teY, 100, sess)
            print("accuracy # before running testing", accuracy)
            # print("accuracy % before running testing", accuracy / int((len(rbf.teY) / batchSize)))
            for i in range(100):
                weightsBefore = sess.run(w_h1)
                cost2 = 0
                for start, end in zip(range(0, len(trX), batchSize), range(batchSize, len(trX)+1, batchSize)):
                    sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
                    cost1 = sess.run(cost, feed_dict={X: trX[start:end], Y: trY[start:end]})
                    #print("Cost", cost1)
                    cost2 += cost1
                #testing
                total_accuracy = []
                # num = tf.Variable(0)
                for start, end in zip(range(0, len(teX), batchSize), range(batchSize, len(teX) + 1, batchSize)):
                    test_batch_accuracy = sess.run(acc, feed_dict={X:teX[start:end], Y:teY[start:end]})
                    total_accuracy.append(test_batch_accuracy)
                    # num = tf.add(num, 1)
                
                test_accuracy_summary, m_accuracy =  sess.run([merged, mean_accuracy], feed_dict={batch_accuracies:total_accuracy})
                summary_writer.add_summary(test_accuracy_summary, i+1)
            k_fold_accuracy += m_accuracy #final accuracy
    k_fold_accuracy = k_fold_accuracy/k_folds
    print("K-fold cross validation accuracy: {}".format(k_fold_accuracy))




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
