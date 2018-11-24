import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.datasets import fetch_lfw_people


def init_weights(s1, s2):
    sigma = math.sqrt(2) * math.sqrt(2 / (s1 + s2))
    return tf.Variable(tf.random_normal([s1, s2], stddev=sigma))


def model(X, w_h1, w_h2, w_o):
    h1 = tf.nn.relu(tf.matmul(X, w_h1))
    h2 = tf.nn.relu(tf.matmul(h1, w_h2))
    return tf.matmul(h2, w_o)


def kfold(data, labels, X, Y, train_op, predict_op):
    accuracy = []
    fold = 1
    for train_index, test_index in KFold(n_splits=10).split(data):
        trX, teX = data[train_index], data[test_index]
        trY, teY = labels[train_index], labels[test_index]

        acc = 0
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(100):
                for start, end in zip(range(0, len(trX), 128), range(128, len(trX) + 1, 128)):
                    sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
                acc = np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX}))

        accuracy.append(acc)
        fold += 1

    return accuracy


def load_data(dataset, pca=False, pca_components=75):
    labels = np.zeros(shape=(dataset['target'].shape[0], dataset['target'].max() + 1))
    for i, j in enumerate(dataset['target']):
        labels[i][j] = 1

    if pca:
        data = PCA(n_components=pca_components, svd_solver='randomized', whiten=True).\
            fit_transform(dataset['data'])
        return data, labels

    data = dataset['data'] / dataset['data'].max()

    return data, labels


def show(accuracy, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([j + 1 for j in range(10)], accuracy)

    plt.xlabel('Fold Iteration')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.show()


dataset = fetch_lfw_people(data_home='.cache/', min_faces_per_person=70)

data, labels = load_data(dataset)

for (hidden_layer1, hidden_layer2) in list([(625, 300), (400, 200)]):
    for learning in list([0.01, 0.05]):

        size_h1 = tf.constant(hidden_layer1, dtype=tf.int32)
        size_h2 = tf.constant(hidden_layer2, dtype=tf.int32)
        X = tf.placeholder("float", [None, data[0].size])
        Y = tf.placeholder("float", [None, len(labels[0])])

        w_h1 = init_weights(data[0].size, hidden_layer1)
        w_h2 = init_weights(hidden_layer1, hidden_layer2)
        w_o = init_weights(hidden_layer2, len(labels[0]))

        py_x = model(X, w_h1, w_h2, w_o)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
        train_op = tf.train.GradientDescentOptimizer(learning).minimize(cost)
        predict_op = tf.argmax(py_x, 1)

        acc = kfold(data, labels, X, Y, train_op, predict_op)
        show(acc, ' '.join(["Accuracy for", str(hidden_layer1), "x",
                           str(hidden_layer2),
                            ", learning rate", str(learning),
                            ", mean", str(np.round(np.array(acc).mean(), 4)),
                            ", stddev", str(np.round(np.array(acc).std(), 4))]))

for (hidden_layer1, hidden_layer2) in list([(625, 300), (400, 200)]):
    for components in list([50, 75, 100]):
        data_pca, labels_pca = load_data(dataset, pca=True, pca_components=components)
        for learning in list([0.01, 0.05]):
            size_h1 = tf.constant(hidden_layer1, dtype=tf.int32)
            size_h2 = tf.constant(hidden_layer2, dtype=tf.int32)
            X = tf.placeholder("float", [None, data_pca[0].size])
            Y = tf.placeholder("float", [None, len(labels_pca[0])])

            w_h1 = init_weights(data_pca[0].size, hidden_layer1)
            w_h2 = init_weights(hidden_layer1, hidden_layer2)
            w_o = init_weights(hidden_layer2, len(labels_pca[0]))

            py_x = model(X, w_h1, w_h2, w_o)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
            train_op = tf.train.GradientDescentOptimizer(learning).minimize(cost)
            predict_op = tf.argmax(py_x, 1)

            acc = kfold(data_pca, labels_pca, X, Y, train_op, predict_op)
            show(acc, ' '.join(["Accuracy for", str(hidden_layer1), "x",
                                str(hidden_layer2), ", PCA components", str(components),
                                ", learning rate", str(learning),
                                ", mean", str(np.array(acc).mean()),
                                ", stddev", str(np.array(acc).std())]))
