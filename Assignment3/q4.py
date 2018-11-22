import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.datasets import fetch_lfw_people

EPOCHS = 100

def load_data(dataset, pca=False, pca_components=75):
    labels = np.zeros(shape=(dataset['target'].shape[0], dataset['target'].max() + 1))
    for i, x in enumerate(dataset['target']):
        labels[i][x] = 1

    if pca:
        # Compute principle components of input data
        data = PCA(n_components=pca_components, svd_solver='randomized', whiten=True).\
            fit_transform(dataset['data'])
    else:
        # Regularize the input data
        data = dataset['data'] / dataset['data'].max()

    return data, labels


class FacialRecognitionNetwork(object):
    def __init__(self, hidden_layer1, hidden_layer2, output_layer, input_size=1850, learning_rate=0.002):
        self.size_h1 = tf.constant(hidden_layer1, dtype=tf.int32)
        self.size_h2 = tf.constant(hidden_layer2, dtype=tf.int32)
        self.X = tf.placeholder("float", [None, input_size])
        self.Y = tf.placeholder("float", [None, output_layer])

        # Initialize weight matrices
        self.w_h1 = self.__init_weights(input_size, hidden_layer1)
        self.w_h2 = self.__init_weights(hidden_layer1, hidden_layer2)
        self.w_o = self.__init_weights(hidden_layer2, output_layer)

        # Initialize the Neural Network model
        self.py_x = self.model()
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.py_x, labels=self.Y))
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)
        self.predict_op = tf.argmax(self.py_x, 1)

        # Cross-Validation attributes
        self.accuracies = []
        self.mean_accuracy = None

    @staticmethod
    def __init_weights(s1, s2):
        sigma = math.sqrt(2) * math.sqrt(2 / (s1 + s2))
        return tf.Variable(tf.random_normal([s1, s2], stddev=sigma))

    def model(self):
        h1 = tf.nn.leaky_relu(tf.matmul(self.X, self.w_h1), alpha=0.2)
        h2 = tf.nn.leaky_relu(tf.matmul(h1, self.w_h2), alpha=0.2)
        return tf.matmul(h2, self.w_o)

    def cross_validate(self, data, labels):
        print("Performing K-Fold cross validation...")
        self.accuracies = []
        fold = 1
        for train_index, test_index in KFold(n_splits=10).split(data):
            trX, teX = data[train_index], data[test_index]
            trY, teY = labels[train_index], labels[test_index]

            acc = 0
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                for i in range(EPOCHS):
                    for start, end in zip(range(0, len(trX), 128), range(128, len(trX) + 1, 128)):
                        sess.run(self.train_op, feed_dict={self.X: trX[start:end], self.Y: trY[start:end]})
                    acc = np.mean(np.argmax(teY, axis=1) == sess.run(self.predict_op, feed_dict={self.X: teX}))

            self.accuracies.append(acc)
            fold += 1

        self.mean_accuracy = np.mean(self.accuracies)

    def plot_accuracies(self, title):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.accuracies)

        plt.xlabel('Fold Iteration')
        plt.ylabel('Accuracy (%)')
        plt.title(title)
        plt.show()


dataset = fetch_lfw_people(data_home='.cache/', min_faces_per_person=70)

data, labels = load_data(dataset)

for components in list(50, 75, 100):
    for learning in list(0.01, 0.05):
        data_pca, labels_pca = load_data(dataset, pca=True, pca_components=components)

        n1 = FacialRecognitionNetwork(200, 50, len(labels[0]), input_size=data[0].size, learning_rate=learning)
        n1_pca = FacialRecognitionNetwork(200, 50, len(labels_pca[0]), input_size=data_pca[0].size,
                                          learning_rate=learning)

        n2 = FacialRecognitionNetwork(160, 60, len(labels[0]), input_size=data[0].size, learning_rate=0.05)
        n2_pca = FacialRecognitionNetwork(160, 60, len(labels_pca[0]), input_size=data_pca[0].size,
                                          learning_rate=learning)

        # Perform K-Fold Cross Validation on all networks
        n1.cross_validate(data, labels)
        n1.plot_accuracies("Facial Recognition Accuracy for 200 x 50 Neural Network")
        print("Mean accuracy of {} achieved.".format(n1.mean_accuracy))

        n1_pca.cross_validate(data_pca, labels_pca)
        n1_pca.plot_accuracies("Facial Recognition Accuracy for 200 x 50 Neural Network with PCA")
        print("Mean accuracy of {} achieved.".format(n1_pca.mean_accuracy))

        n2.cross_validate(data, labels)
        n2.plot_accuracies("Facial Recognition Accuracy for 160 x 60 Neural Network")
        print("Mean accuracy of {} achieved.".format(n2.mean_accuracy))

        n2_pca.cross_validate(data_pca, labels_pca)
        n2_pca.plot_accuracies("Facial Recognition Accuracy for 160 x 60 Neural Network with PCA")
        print("Mean accuracy of {} achieved.".format(n2_pca.mean_accuracy))