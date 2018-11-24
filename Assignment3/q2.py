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
centroids = 75


def model(X, w, centroid, numCentroids, b, keep_prob=0.5, use_dropout = True):
    #with tf.Session() as sess:
    #tf.global_variables_initializer().run()
    b = tf.transpose(b)
    #print("stack", sess.run(tf.tile(tf.expand_dims(X, 1), [1, 2, 1]), feed_dict={X: [[0, 0], [1, 1]]}))
    #print("centroid", sess.run(tf.reshape(tf.tile(centroid, [2, 1]), [2, 2, 2]), feed_dict={X: [[0, 0], [1, 1]]}))

    x1 = tf.to_float(tf.tile(tf.expand_dims(X, 1), [1, numCentroids, 1]))

    # dropout = tf.nn.dropout(w, keep_prob=keep_prob)

    centroid1 = tf.to_float(tf.reshape(tf.tile(centroid, [tf.shape(X)[0], 1]), [tf.shape(X)[0],
                                                                                numCentroids,
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
    if use_dropout:
        dropout = tf.nn.dropout(w, keep_prob=keep_prob)
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

        # print('shape of labels:', self.labels.shape)
        # self.data_separated = [[] for i in range(10)]
        # self.labels_separated = [[] for i in range(10)]

        # print(self.labels_separated)
        # print("length", len(self.data))
        # for i in range(len(self.data)):
        #     data_label = np.where(self.labels[i]==1)[0][0] #find index of 1 in one_hot encoded label
        #     self.data_separated[data_label].append(self.data[i]) #use label as index for data_separated
        #     self.labels_separated[data_label].append(self.labels[i])
        # for i in range(10):
        #     print(len(self.data_separated[i]), len(self.labels_separated[i]))
        # self.labels_separated = np.asarray(self.labels_separated)
        
        # self.ordered_data = self.data_separated[0]
        # self.ordered_labels = self.labels_separated[0]
        # self.ordered_labels = [self.ordered_labels.extend(self.labels_separated[i]) for i in range(len(self.labels_separated))]
        # print('len', len(self.ordered_labels))
        # # self.ordered_labels = self.ordered_labels.flatten()
        # for i in range(1,10):
        #     self.ordered_data = np.append(self.ordered_data, self.data_separated[i], axis=0)
        #     # print(len(self.ordered_labels))
        #     # print(self.labels_separated.shape
        #     # self.ordered_labels = np.append(self.ordered_labels, self.labels_separated[i])
        # # self.data, self.labels = shuffle(self.data, self.labels)
        # print(self.ordered_data.shape)
        # # print(self.ordered_labels.shape)

    def kmean(self):
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter('logs/kmean/', graph=sess.graph)

            for k in range(20, 150, 2):
                avg = 0
                #average the mean over 3 runs
                for _ in range(3):
                    kmeans = kmeans = sk.KMeans(n_clusters=k, init='random', n_init=1).fit(self.data)
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
                        
                # for _ in range(0, 10, 1):
                #     kmeans = sk.KMeans(n_clusters=k, init='random', n_init=1).fit(self.data)
                #     means = kmeans.cluster_centers_

                #     avg += kmeans.inertia_
                #     print(k, kmeans.inertia_)

                # kmeans = sk.KMeans(n_clusters=k, n_init=1).fit(self.data)
                # print(k, "using best fit", kmeans.inertia_)


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
k_folds = 5
fold = 0
epochs = 50
rbf = mnistDataset()
# rbf.kmean() #Uncomment this line to get the kmeans graph, we see that the elbow is at around k=70 to k=80
#As such, we will pick k=75 
#Modify these numbers to answer question 3 and 4
kcentroids = [50,60,70,80,90] 
keep_probs = [1.0]
kcentroid_accuracies = []
for numCentroids in kcentroids:
    c = rbf.getCentroids(k=numCentroids)

    for keep_prob in keep_probs:
        #try with dropout and without
        tf.reset_default_graph()
        w_h1 = init_weights([numCentroids, output_size])
        X = tf.placeholder("float", shape=[None, input_size])
        Y = tf.placeholder("float", shape=[None, output_size])
        py_x = model(X, w_h1, centroid=c, numCentroids=numCentroids,
                    b=rbf.getBetas(c), keep_prob=keep_prob)

        test_pred = model(X, w_h1, centroid=c,numCentroids=numCentroids,
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
        # train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

        #summaries
        tf.summary.scalar('accuracy', mean_accuracy)

        batchSize = 128
        testSize = 100

        for train_index, test_index in KFold(n_splits=k_folds).split(rbf.data):
            fold += 1
            print(train_index, test_index)
            trX, teX = rbf.data[train_index], rbf.data[test_index]
            trY, teY = rbf.labels[train_index], rbf.labels[test_index]
            print(len(trX), len(trY))
            result_dir = './logs/attempt_{}/fold_{}_keep_prob_{}_centroids_{}/'.format(attempt,fold, int(keep_prob*100), numCentroids)
            with tf.Session() as sess:
                merged = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter(result_dir, graph=sess.graph)
                tf.global_variables_initializer().run()
                
                # print("accuracy % before running testing", accuracy / int((len(rbf.teY) / batchSize)))
                for i in range(50):
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
                    print("Epoch : {}, Test Acc: {}".format(i+1, m_accuracy))
                k_fold_accuracy += m_accuracy #final accuracy
        k_fold_accuracy = k_fold_accuracy/k_folds
        kcentroid_accuracies.append(k_fold_accuracy)
        with tf.Session() as sess:
            sw = tf.summary.FileWriter('logs/attempt_{}/k_hidden_neurons'.format(attempt), graph=sess.graph)
            sw.add_summary(tf.Summary(value=[
                tf.Summary.Value(tag="Accuracy", simple_value=k_fold_accuracy),
            ]), numCentroids)

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
