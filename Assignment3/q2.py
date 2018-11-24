import sklearn.cluster as sk
from sklearn.utils import shuffle
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from sklearn.model_selection import KFold
import os


def init_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=1.0))


input_size = 784
output_size = 10
centroids = 75


def model(X, w, centroid, numCentroids, b, keep_prob=0.5, use_dropout=True):
    #X has shape [None, 784]
    # b = tf.transpose(b)
    # print(b)
    # x1 = tf.to_float(tf.tile(tf.expand_dims(X, 1), [1, numCentroids, 1]))
    # centroid1 = tf.to_float(tf.reshape(tf.tile(centroid, [tf.shape(X)[0], 1]), [tf.shape(X)[0],
    #                                                                             numCentroids,
    #                                                                             input_size]))
    #
    # dist = tf.square(tf.norm(tf.subtract(x1, centroid1), axis=-1))
    # print(dist)
    # negative = tf.to_float(tf.negative(b))
    # beta = tf.multiply(negative, dist)
    # exponent = tf.exp(beta)
    # print(exponent)
    # with tf.Session() as sess:
    #     print(sess.run(X))
    #     print(sess.run(centroid))
    # print('X',X)
    # print('centroid,', centroid)
    # subtract = tf.subtract(X, centroid)
    # print(subtract)
    sum1 = tf.reduce_sum([X, tf.negative(centroid)], axis=0)
    print(sum1)
    norm = tf.norm(sum1, axis=1, keepdims=True)
    print(norm)
    dist = tf.square(norm)
    print(dist)
    negative = tf.to_float(tf.negative(b))
    beta = tf.multiply(negative, dist)
    exponent = tf.exp(beta)
    print('b,', b)

    if use_dropout:
        dropout = tf.nn.dropout(w, keep_prob=keep_prob)
        print(dist)
        print(dropout)
        print(exponent)
        print(b)
        return tf.matmul(exponent, dropout)
    else:
        return tf.matmul(exponent, w)


class mnistDataset:
    def __init__(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        trX1, trY1, teX1, teY1 = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

        self.data = np.append(trX1, teX1, axis=0)
        self.labels = np.append(trY1, teY1, axis=0)
        self.data, self.labels = shuffle(self.data, self.labels)

        # normalize data
        for i in range(len(self.data)):
            image_vector = self.data[i]
            self.data[i] = [j/255.0 for j in image_vector]

    def kmean(self):
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter('logs/kmean/', graph=sess.graph)

            for k in range(20, 150, 2):
                avg = 0
                # average the mean over 3 runs
                for _ in range(3):
                    kmeans = sk.KMeans(n_clusters=k, init='random', n_init=1).fit(self.data)
                    means = kmeans.cluster_centers_

                    sek = 0
                    for x in self.data:
                        best = float("inf")
                        index = -1
                        for y in range(len(means)):
                            d = np.linalg.norm(x-means[y])
                            if d < best:
                                best = d
                                index = y
                        sek += best
                    avg += sek
                    print(k, sek, index, best)
                    
                summary_writer.add_summary(tf.Summary(value=[
                    tf.Summary.Value(tag="objective function", simple_value=avg/3.0),
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
        attempt += 1
    else:
        break

k_fold_accuracy = 0
k_folds = 5
fold = 0
epochs = 50
rbf = mnistDataset()
# rbf.kmean() #Uncomment this line to get the kmeans graph, we see that the elbow is at around k=70 to k=80
#As such, we will pick k=75 
#Modify these numbers to answer question 3 and 4
kcentroids = [70] 
keep_probs = [0.85, 1.0]
kcentroid_accuracies = []
for numCentroids in kcentroids:
    c = rbf.getCentroids(k=numCentroids)

    for keep_prob in keep_probs:
        # try with dropout and without
        tf.reset_default_graph()
        w_h1 = init_weights([numCentroids, output_size])
        X = tf.placeholder("float", shape=[None, input_size])
        Y = tf.placeholder("float", shape=[None, output_size])
        py_x = model(X, w_h1, centroid=c, numCentroids=numCentroids,
                     b=rbf.getBetas(c), keep_prob=keep_prob)

        test_pred = model(X, w_h1, centroid=c, numCentroids=numCentroids,
                          b=rbf.getBetas(c), keep_prob=keep_prob, use_dropout=False)

        predict_op = tf.argmax(test_pred, 1)
        y_true_cls = tf.argmax(Y, dimension=1)
        batch_accuracies = tf.placeholder("float", [None])
        num_batches = tf.placeholder("float")
        with tf.name_scope('accuracy'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) #for training so use py_x
            correct_prediction = tf.equal(predict_op, y_true_cls)
            acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            mean_accuracy = tf.reduce_mean(tf.cast(batch_accuracies, tf.float32))
        
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        # summaries
        tf.summary.scalar('accuracy', mean_accuracy)

        batchSize = 70
        
        k_fold_accuracy = 0
        for train_index, test_index in KFold(n_splits=k_folds).split(rbf.data):
            fold += 1
            print(train_index, test_index)
            trX, teX = rbf.data[train_index], rbf.data[test_index]
            trY, teY = rbf.labels[train_index], rbf.labels[test_index]
            print(len(trX), len(trY))
            result_dir = './logs/attempt_{}/fold_{}_keep_prob_{}_centroids_{}/'.format(attempt,
                                                                                       fold,
                                                                                       int(keep_prob*100),
                                                                                       numCentroids)
            with tf.Session() as sess:
                merged = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter(result_dir, graph=sess.graph)
                tf.global_variables_initializer().run()
                
                # print("accuracy % before running testing", accuracy / int((len(rbf.teY) / batchSize)))
                for i in range(200):
                    cost2 = 0
                    for start, end in zip(range(0, len(trX), batchSize), range(batchSize, len(trX)+1, batchSize)):
                        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
                        # cost1 = sess.run(cost, feed_dict={X: trX[start:end], Y: trY[start:end]})
                        #print("Cost", cost1)
                        # cost2 += cost1
                    # testing
                    total_accuracy = []
                    # num = tf.Variable(0)
                    for start, end in zip(range(0, len(teX), batchSize), range(batchSize, len(teX) + 1, batchSize)):
                        test_batch_accuracy = sess.run(acc, feed_dict={X: teX[start:end], Y: teY[start:end]})
                        # print('accuracy:',test_batch_accuracy)
                        total_accuracy.append(test_batch_accuracy)
                        # num = tf.add(num, 1)
                    
                    test_accuracy_summary, m_accuracy = sess.run([merged, mean_accuracy],
                                                                 feed_dict={batch_accuracies: total_accuracy})

                    summary_writer.add_summary(test_accuracy_summary, i+1)
                    print("Epoch : {}, Test Acc: {}".format(i+1, m_accuracy))
                k_fold_accuracy += m_accuracy #final accuracy

        k_fold_accuracy = k_fold_accuracy/k_folds
        kcentroid_accuracies.append(k_fold_accuracy)
        with tf.Session() as sess:
            # sw = tf.summary.FileWriter('logs/attempt_{}/k_hidden_neurons'.format(attempt), graph=sess.graph)
            sw = tf.summary.FileWriter('logs/attempt_{}/k_dropout'.format(attempt), graph=sess.graph)
            sw.add_summary(tf.Summary(value=[
                tf.Summary.Value(tag="Accuracy", simple_value=k_fold_accuracy),
            ]), keep_probs)

        print("K-fold cross validation accuracy: {}".format(k_fold_accuracy))


