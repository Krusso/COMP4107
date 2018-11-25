import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from minisom import MiniSom
from random import shuffle

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import tensorflow as tf
from scipy.spatial import Voronoi


def voronoi_finite_polygons_2d(vor, radius=None):
    # https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


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
    plt.axis([0, x_dim, 0, y_dim])
    plt.title(title)
    plt.show()


#Part a)
dataset = get_data()
x_dim = 20
y_dim = 20
input_size = 784
sigma = .9
lr = .25

som = MiniSom(x_dim, y_dim, input_size, sigma=sigma, learning_rate=lr)
plot_som('SOM - before training', som)
som.train_random([i[0][0] for i in dataset], 5000)
plot_som('SOM - after training %s epochs' % 5000, som)

# Part b)
for k in [2, 4, 6, 8, 10, 12, 14, 16, 18]:
    # PCA uses SVD
    pca = PCA(n_components=2).fit_transform(np.array([np.array(scale(x[0][0])) for x in dataset]))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pca)

    vor = Voronoi(kmeans.cluster_centers_)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    # colorize
    for region in regions:
        polygon = vertices[region]
        plt.fill(*zip(*polygon), alpha=0.4)

    # # https://en.wikipedia.org/wiki/Voronoi_diagram
    legend = []
    for label, color in [(1, 'purple'), (5, 'blue')]:
        a = np.array([pca[i] for i in range(len(dataset)) if dataset[i][1] != label])
        plt.plot(a[:, 0], a[:, 1], 'k.', markersize=10, color=color)
        legend.append(patches.Patch(color=color, label=str(label)))

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                s=1, color='w', zorder=10, marker='*', linewidth=10)

    plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
    plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)

    plt.legend(handles=legend)
    plt.show()


    # plt.figure(figsize=(5, 5))
    #
    # x_min, x_max = pca[:, 0].min() - 1, pca[:, 0].max() + 1
    # y_min, y_max = pca[:, 1].min() - 1, pca[:, 1].max() + 1
    #
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, .1), np.arange(y_min, y_max, .1))
    # bounds = [xx.min(), xx.max(), yy.min(), yy.max()]
    # predictions = kmeans.predict(np.vstack((xx.flatten(), yy.flatten())).T)
    #
    # plt.imshow(predictions.reshape(xx.shape), extent=bounds, cmap=plt.cm.Dark2, origin='lower')
    #
    # legend = []
    # # https://en.wikipedia.org/wiki/Voronoi_diagram
    # for label, color in [(1, 'purple'), (5, 'blue')]:
    #     a = np.array([pca[i] for i in range(len(dataset)) if dataset[i][1] != label])
    #     plt.plot(a[:, 0], a[:, 1], 'k.', markersize=10, color=color)
    #     legend.append(patches.Patch(color=color, label=str(label)))
    #
    # centroids = kmeans.cluster_centers_
    # plt.scatter(centroids[:, 0], centroids[:, 1], s=1, color='w', zorder=10, marker='*', linewidth=10)
    #
    # plt.title('K-means with PCA')
    # plt.legend(handles=legend)
    # plt.xticks([])
    # plt.yticks([])
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)
    # plt.show()
