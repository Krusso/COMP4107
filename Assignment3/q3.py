import numpy as np
import matplotlib.pyplot as plt

from minisom import MiniSom
from random import shuffle

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import tensorflow as tf


# Load dataset
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


# Plots what the current SOM looks like
def plot_som(title, som):
    plt.figure(figsize=(5, 5))
    for index, item in enumerate(dataset):
        image, label = item
        i, j = som.winner(image)
        plt.text(i, j, str(int(label)), color=plt.cm.Dark2(label / 5.), fontdict={'size': 12})
    plt.axis([0, 20, 0, 20])
    plt.title(title)
    plt.show()


#Part a)
dataset = get_data()

print("Showing som before training")
som = MiniSom(20, 20, 784, sigma=.9, learning_rate=.25)
plot_som('SOM before training', som)
print("training som")
som.train_random([i[0][0] for i in dataset], 5000)
print("Showing som after training")
plot_som('SOM after training %s epochs' % 5000, som)

# Part b)
for k in [2, 4, 6, 8, 10, 12, 14, 16, 18]:
    # PCA uses SVD
    pca = PCA(n_components=2).fit_transform(np.array([np.array(scale(x[0][0])) for x in dataset]))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pca)

    x_min, x_max = pca[:, 0].min() - 1, pca[:, 0].max() + 1
    y_min, y_max = pca[:, 1].min() - 1, pca[:, 1].max() + 1

    x1, y1 = np.meshgrid(np.linspace(x_min, x_max, (x_max - x_min)/.1),
                         np.linspace(y_min, y_max, (x_max - x_min)/.1))
    bounds = [x1.min(), x1.max(), y1.min(), y1.max()]
    predictions = kmeans.predict(np.vstack((x1.flatten(), y1.flatten())).T)

    # plot regions
    plt.imshow(predictions.reshape(x1.shape), extent=bounds, origin='lower')

    legend = []
    # https://en.wikipedia.org/wiki/Voronoi_diagram
    # plot mnist data
    for label, color in [(1, 'purple'), (5, 'blue')]:
        a = np.array([pca[i] for i in range(len(dataset)) if dataset[i][1] != label])
        plt.plot(a[:, 0], a[:, 1], 'ro', markersize=2, color=color, label=str(label))

    # plot white centroids on top of everything else
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='w', zorder=100)

    plt.title('K-means visualized using PCA')
    plt.legend()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()
