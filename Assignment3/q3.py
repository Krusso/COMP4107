import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from minisom import MiniSom
from random import shuffle

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = np.concatenate((x_train, x_test))
y_train = np.concatenate((y_train, y_test))

dataset = []
for x in range(len(x_train)):
    if y_train[x] == 1 or y_train[x] == 5:
        dataset.append((x_train[x].reshape([1, 784]), y_train[x]))

print(len(dataset))

shuffle(dataset)

x_dim = 20
y_dim = 20
input_len = 784
sigma = .9 # spread of neighbourhood function
learning_rate = .25

som = MiniSom(x_dim, y_dim, input_len, sigma=sigma, learning_rate=learning_rate)


def som_demo(title):
    plt.figure(figsize=(5, 5))

    i = 0
    for index, item in enumerate(dataset):
        i += 1
        if i % 100 == 0:
            print("Training at", i)
        image, label = item
        i, j = som.winner(image)
        plt.text(i, j, str(label), color=plt.cm.Dark2(label / 5.), fontdict={'size': 12})
    plt.axis([0, x_dim, 0, y_dim])
    plt.title(title)
    plt.show()


som_demo('SOM - before training')
epochs = 1500
print([i[0][0] for i in dataset[:2]])
som.train_random([i[0] for i in dataset[:1024]], epochs)
som_demo('SOM - after training %s epochs' % epochs)


def min_max(np_arr):
    return [np_arr.min() - 1, np_arr.max() + 1]


rd = PCA(n_components=2).fit_transform([scale(x[0]) for x in dataset[:1024]])
kmeans = KMeans(n_clusters=2)
kmeans.fit(rd)

plt.figure(figsize=(5, 5))

x_min, x_max = min_max(rd[:,0])
y_min, y_max = min_max(rd[:,1])

xx, yy = np.meshgrid(np.arange(x_min, x_max, .1), np.arange(y_min, y_max, .1))
bounds = [xx.min(), xx.max(), yy.min(), yy.max()]
predictions = kmeans.predict(zip(xx.flatten(), yy.flatten()))
plt.imshow(predictions.reshape(xx.shape), extent=bounds, cmap=plt.cm.Accent_r, origin='lower')

legend = []
for label, color in [(1, 'orange'), (5, 'purple')]:
    _ = np.array([rd[i] for i in range(1024) if dataset[i][1] is label])
    plt.plot(_[:, 0], _[:, 1], 'k.', markersize=10, color=color)
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