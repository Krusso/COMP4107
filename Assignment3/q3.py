import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from minisom import MiniSom
from random import shuffle

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import tensorflow as tf

#Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.concatenate((x_train, x_test)).astype(np.float)
y_train = np.concatenate((y_train, y_test)).astype(np.float)
dataset = []
for x in range(len(x_train)):
    if y_train[x] == 1 or y_train[x] == 5:
        dataset.append((x_train[x].reshape([1, 784]), y_train[x]))
shuffle(dataset)

x_dim = 20
y_dim = 20
input_len = 784
sigma = .9 # spread of neighbourhood function
learning_rate = .25

som = MiniSom(x_dim, y_dim, input_len, sigma=sigma, learning_rate=learning_rate)

#Document the dimensions of the SOM computed and the learning parameters
def som_demo(title):
    plt.figure(figsize=(5, 5))

    for index, item in enumerate(dataset):
        image, label = item
        i, j = som.winner(image)
        plt.text(i, j, str(int(label)), color=plt.cm.Dark2(label / 5.), fontdict={'size': 12})
    plt.axis([0, x_dim, 0, y_dim])
    plt.title(title)
    plt.show()

som_demo('SOM - before training')
epochs = 1500
som.train_random([i[0][0] for i in dataset], epochs)
som_demo('SOM - after training %s epochs' % epochs)


def min_max(np_arr):
    return [np_arr.min() - 1, np_arr.max() + 1]


for k in range(2, 10):
    # PCA uses SVD
    rd = PCA(n_components=2).fit_transform(np.array([np.array(scale(x[0][0])) for x in dataset]))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(rd)

    plt.figure(figsize=(5, 5))

    x_min, x_max = min_max(rd[:,0])
    y_min, y_max = min_max(rd[:,1])

    xx, yy = np.meshgrid(np.arange(x_min, x_max, .1), np.arange(y_min, y_max, .1))
    bounds = [xx.min(), xx.max(), yy.min(), yy.max()]
    predictions = kmeans.predict(np.vstack((xx.flatten(), yy.flatten())).T)
    plt.imshow(predictions.reshape(xx.shape), extent=bounds, cmap=plt.cm.Accent_r, origin='lower')

    legend = []
    # https://en.wikipedia.org/wiki/Voronoi_diagram
    for label, color in [(1, 'orange'), (5, 'purple')]:
        a = np.array([rd[i] for i in range(len(dataset)) if dataset[i][1] != label])
        plt.plot(a[:, 0], a[:, 1], 'k.', markersize=10, color=color)
        legend.append(patches.Patch(color=color, label=str(label)))

    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], s=1, color='w', zorder=10, marker='*', linewidth=10)

    plt.title('K-means Clustering on PCA Reduced Mnist')
    plt.legend(handles=legend)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()
