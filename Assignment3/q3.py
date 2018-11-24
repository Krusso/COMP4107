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
def get_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.concatenate((x_train, x_test)).astype(np.float)
    y_train = np.concatenate((y_train, y_test)).astype(np.float)
    dataset = []
    for x in range(len(x_train)):
            if y_train[x] == 1 or y_train[x] == 5:
                    dataset.append((x_train[x].reshape([1, 784]), y_train[x]))
            shuffle(dataset)
    return dataset  

#Plots what the current SOM looks like
def plot_som(title, som):
    plt.figure(figsize=(5, 5))
    for index, item in enumerate(dataset):
        image, label = item
        i, j = som.winner(image)
        plt.text(i, j, str(int(label)), color=plt.cm.Dark2(label / 5.), fontdict={'size': 12})
    plt.axis([0, x_dim, 0, y_dim])
    plt.title(title)
    plt.show()

def min_max(np_arr):
    return [np_arr.min() - 1, np_arr.max() + 1]

#################################################
#Part a)
dataset = get_data()
x_dim = 20
y_dim = 20
input_size = 784
sigma = .9 # spread of neighbourhood function
lr = .25

som = MiniSom(x_dim, y_dim, input_size, sigma=sigma, learning_rate=lr)
plot_som('SOM - before training', som)
epochs = 1500
som.train_random([i[0][0] for i in dataset], epochs)
plot_som('SOM - after training %s epochs' % epochs, som)

#Part b)
for k in [2,4,6,8,10,12,14,16,18]:
    # PCA uses SVD
    pca = PCA(n_components=2).fit_transform(np.array([np.array(scale(x[0][0])) for x in dataset]))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pca)

    plt.figure(figsize=(5, 5))

    x_min, x_max = min_max(pca[:,0])
    y_min, y_max = min_max(pca[:,1])

    xx, yy = np.meshgrid(np.arange(x_min, x_max, .1), np.arange(y_min, y_max, .1))
    bounds = [xx.min(), xx.max(), yy.min(), yy.max()]
    predictions = kmeans.predict(np.vstack((xx.flatten(), yy.flatten())).T)
#     print(plt.cm.)
    plt.imshow(predictions.reshape(xx.shape), extent=bounds, cmap=plt.cm.Dark2, origin='lower')

    legend = []
    # https://en.wikipedia.org/wiki/Voronoi_diagram
    for label, color in [(1, 'purple'), (5, 'blue')]:
        a = np.array([pca[i] for i in range(len(dataset)) if dataset[i][1] != label])
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
