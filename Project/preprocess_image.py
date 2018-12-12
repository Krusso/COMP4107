import tensorflow as tf
import sys
import os
import argparse
import numpy as np
import cv2
import glob
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

import imblearn.over_sampling as imb

FLAGS = None


def visualize_scatter(data_2d, label_ids, id_to_label_dict, figsize=(5, 5)):
    plt.figure(figsize=figsize)
    plt.grid()

    nb_classes = len(np.unique(label_ids))

    for label_id in np.unique(label_ids):
        print("Class", label_id, "length", np.array(np.where(label_ids == label_id)).shape)
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    color=plt.cm.Set1(label_id / float(nb_classes)),
                    linewidth='1',
                    alpha=0.8,
                    label=id_to_label_dict[label_id])

    plt.legend(loc='best')

    plt.show()


def show_tnse(data, labels, height, width):
    id_to_label_dict = {
        0: 'airplane',
        1: 'car',
        2: 'cat',
        3: 'dog',
        4: 'flower',
        5: 'fruit',
        6: 'motorbike',
        7: 'person'
    }

    data = data.flatten().reshape(len(data), height * width * 3)

    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(data)
    pca_result_scaled = StandardScaler().fit_transform(pca_result)
    visualize_scatter(pca_result_scaled, np.argmax(labels, axis=1), id_to_label_dict)

    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(data)

    print('Cumulative explained variation for 180 principal components: {}'.
          format(np.sum(pca.explained_variance_ratio_)))

    tsne = TSNE(n_components=2, perplexity=40.0, n_iter=7000)
    tsne_result = tsne.fit_transform(pca_result)
    tsne_result_scaled = StandardScaler().fit_transform(tsne_result)

    visualize_scatter(tsne_result_scaled, np.argmax(labels, axis=1), id_to_label_dict)

    tsne = TSNE(n_components=3, perplexity=40.0)
    tsne_result = tsne.fit_transform(pca_result)
    tsne_result_scaled = StandardScaler().fit_transform(tsne_result)
    label_ids = np.argmax(labels, axis=1)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    plt.grid()

    nb_classes = len(np.unique(label_ids))

    for label_id in np.unique(label_ids):
        ax.scatter(tsne_result_scaled[np.where(label_ids == label_id), 0],
                   tsne_result_scaled[np.where(label_ids == label_id), 1],
                   tsne_result_scaled[np.where(label_ids == label_id), 2],
                   alpha=0.8,
                   color=plt.cm.Set1(label_id / float(nb_classes)),
                   marker='o',
                   label=id_to_label_dict[label_id])
    ax.legend(loc='best')
    ax.view_init(25, 45)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_zlim(-2.5, 2.5)

    plt.show()


# short for quick testing, will only deal with several hundred images not the entire dataset
# cropAndPad if false use tf.image.resize_images if true then use tf.image_image_with_crop_or_pad
def natural_images(path='./natural_images', width=64, height=64, cropAndPad=False, short=False):
    files = [('airplane', [1, 0, 0, 0, 0, 0, 0, 0]),
             ('car', [0, 1, 0, 0, 0, 0, 0, 0]),
             ('cat', [0, 0, 1, 0, 0, 0, 0, 0]),
             ('dog', [0, 0, 0, 1, 0, 0, 0, 0]),
             ('flower', [0, 0, 0, 0, 1, 0, 0, 0]),
             ('fruit', [0, 0, 0, 0, 0, 1, 0, 0]),
             ('motorbike', [0, 0, 0, 0, 0, 0, 1, 0]),
             ('person', [0, 0, 0, 0, 0, 0, 0, 1])]

    # dataset contains 6899 images
    dataset = np.ndarray(shape=(6899, height, width, 3),
                         dtype=np.float32)

    labels = np.ndarray(shape=(6899, 8),
                        dtype=np.float32)

    # with tf.Session() as sess:
    i = 0
    for file, label in files:
        filepath = os.path.join(path, file)
        filepath += '/*.jpg'
        print(filepath)

        for filename in glob.glob(filepath):

            im = cv2.imread(filename, cv2.IMREAD_COLOR)
            im = im.astype(np.float32)
            im = np.multiply(im, 1.0 / 255.0)
            h, w, c = im.shape
            # cv2.imshow('original',im)
            # print(im.shape)
            # print(height, width)
            if FLAGS.method == 'cp':
                # im = tf.image.resize_image_with_crop_or_pad(im, [height, width])
                # im = cv2.resize(im, [height, width], interpolation=cv2.INTER_AREA)
                if w > width:  # if w is greater than resize width, crop it about the center.
                    im = im[:, w // 2 - width // 2: w // 2 + width // 2]
                else:  # if w is smaller than the resize width, pad it with 0 about the center
                    num_padding = width - w
                    left_pad = num_padding // 2
                    right_pad = num_padding - left_pad
                    im = cv2.copyMakeBorder(im, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)

                # print(im)
                if h > height:
                    im = im[h // 2 - height // 2: h // 2 + height // 2]
                else:
                    num_padding = height - h
                    top_pad = num_padding // 2
                    bot_pad = num_padding - top_pad
                    im = cv2.copyMakeBorder(im, top_pad, bot_pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
                # print(im.shape)
            elif FLAGS.method == 'r':
                im = cv2.resize(im, (height, width), interpolation=cv2.INTER_NEAREST)
                
                # im = tf.image.resize_images(im, [height, width], align_corners=True)
            # cv2.imshow('image',im)
            # cv2.waitKey(0) 
            dataset[i] = im
            labels[i] = label
            i += 1

            if i % 100 == 0 and i > 0:
                print(i, "read")
                if short:
                    break

    print("Done reading images")

    print("Starting to generate synthetic data")

    # TODO: @Michael add SMOTE/ADANYS here

    # Smote

    print('Flattening image dataset to sample')
    dataset = dataset.flatten().reshape(6899, height * width * 3)
    print('Fitting samples...')

    unhotlabels = np.argmax(labels, axis=1)

    unique, counts = np.unique(unhotlabels, return_counts=True)
    print("count before", dict(zip(unique, counts)))

    print("Using:", FLAGS.sampling)
    if FLAGS.sampling == 'smote':
        dataset, labels = imb.SMOTE(n_jobs=2).fit_sample(dataset, unhotlabels)
    elif FLAGS.sampling == 'adasyn':
        dataset, labels = imb.ADASYN(sampling_strategy={
            0: 1400,
            1: 1400,
            2: 1400,
            3: 1400,
            4: 1400,
            5: 1400,
            6: 1400,
            7: 1400,
        }, n_jobs=4).fit_sample(dataset, unhotlabels)
    elif FLAGS.sampling == 'none':
        dataset, labels = dataset, unhotlabels

    unique, counts = np.unique(labels, return_counts=True)
    print("count after", dict(zip(unique, counts)))

    # hot encoding
    n_values = np.max(labels) + 1
    labels = np.eye(n_values)[labels]

    dataset = dataset.reshape(dataset.shape[0], height, width, 3)

    print("Finished generating new data")

    # trainData, trainLabel, testData, testLabel
    # TODO: @Michael/@Krystian shuffle the data before returning it and set the bounds for training/testing correctly

    # num examples
    num_examples = dataset.shape[0]
    kfold = 10
    split_index = num_examples//kfold
    if short:
        return dataset[0:400], labels[0:400], \
        dataset[400:600], labels[400:600]

    dataset, labels = shuffle(dataset, labels)
    return dataset[split_index:], labels[split_index:], \
        dataset[:split_index], labels[:split_index]


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _write_example(image, label):
    """
        Args: image - a numpy array of the image
              label - the integer value representing the image label
        Return: tf.train.Example
    """
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image_raw': _bytes_feature(image.tostring()),
                'label': _int64_feature(label)
            }))
    return example


def salt_pepper(image):
    row, col, ch = image.shape
    s_vs_p = 0.5
    amount = 0.05
    out = np.copy(image)
    x = np.random.random_integers(0, row - 1, int(amount * row * col))
    y = np.random.random_integers(0, col - 1, int(amount * row * col))
    n = int(len(x) * s_vs_p)

    # salt
    for i, j in zip(x[:n], y[:n]):
        # print(i, j)
        out[i][j] = 1
    # pepper
    for i, j in zip(x[n:], y[n:]):
        # print(i, j)
        out[i][j] = 0
    return out

def random_hsv(image):
    """
        input: cv2 image type of BGR
        output: a randomly adjusted hue, saturation and brightness of input image
    """

    d_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV, )
    random_adjustment = float(np.random.random_integers(20,160))
    d_img[:,:,0] += random_adjustment
    # print("HUE BEFORE",d_img[:,:,0])
    d_img[:,:,0] %= 180
    # print("HUE AFTER", d_img[:,:,0])
    random_adjustment = float(np.random.random_integers(20,225))
    # print("b d_img 1", d_img[:,:,1])
    d_img[:,:,1] *=  255
    d_img[:,:,1] += random_adjustment
    # print("m d_img 1", d_img[:,:,1])
    d_img[:,:,1] %= 255
    d_img[:,:,1] /= 255.0
    # print("a d_img 1", d_img[:,:,1])
    d_img[:,:,2] +=  float(np.random.random_integers(-15,15)/100)
    d_img = cv2.cvtColor(d_img, cv2.COLOR_HSV2BGR)
    return d_img

def modify_image(trX, trY):
    """
        Distort each image by randomly rotating 90 degrees and adding pepper&salt to the image
        input: image and labels
        output: shuffled modified images + original images, labels
    """
    distorted = []
    labels = []
    for i in range(len(trX)):
        # cv2.imshow('original', trX[i])
        for j in range(1,4):
            
            d_img = random_hsv(trX[i])
            d_img = salt_pepper(np.rot90(d_img, j))
            distorted.append(d_img)
            labels.append(trY[i])
            # cv2.imshow('distorted {}'.format(j), d_img)
        # cv2.waitKey(0)
    distorted.extend(trX)
    labels.extend(trY)
    return shuffle(distorted, labels)


def main(unused_argv):
    DISTORTED_IMAGE_DIR = ".\distorted_images"
    if not os.path.exists(DISTORTED_IMAGE_DIR):
        os.makedirs(DISTORTED_IMAGE_DIR)

    if FLAGS.method == 'cp':
        cropAndPad = True
    else:
        cropAndPad = False

    print("Read natural images dataset")
    trX, trY, teX, teY = natural_images(path='./natural_images', height=FLAGS.h, width=FLAGS.w, cropAndPad=cropAndPad, short=False)
    # Different sizes we can try for the height/width of the modified images
    # trX, trY, teX, teY = cifar10(path='./natural_images', height=32, width=32, cropAndPad=True)
    # trX, trY, teX, teY = cifar10(path='./natural_images', height=64, width=64, cropAndPad=False)
    # trX, trY, teX, teY = cifar10(path='./natural_images', height=64, width=64, cropAndPad=True)
    # trX, trY, teX, teY = cifar10(path='./natural_images', height=64, width=128, cropAndPad=False)
    # trX, trY, teX, teY = cifar10(path='./natural_images', height=32, width=128, cropAndPad=True)

    train_file = os.path.join(DISTORTED_IMAGE_DIR,
                              'train_set_h{}w{}_{}_{}.tfrecords'.format(FLAGS.h, FLAGS.w, FLAGS.method, FLAGS.sampling))
    test_file = os.path.join(DISTORTED_IMAGE_DIR,
                             'test_set_h{}w{}_{}_{}.tfrecords'.format(FLAGS.h, FLAGS.w, FLAGS.method, FLAGS.sampling))

    # TODO: @Michael get the data augmentation call back into the pipeline
    trX, trY = modify_image(trX, trY)
    print("Completed Modification of Images")

    n = len(trX)
    train_set = []
    test_set = []
    print("Converting modified dataset to tf Examples...")
    print("Length training X", len(trX))
    print("Length training Y", len(trY))
    for i in range(n):
        # convert to tf examples, labels one-hot encoding converted to integer (later parsing will need to convert back)
        example = _write_example(trX[i], np.where(trY[i] == 1)[0][0])
        train_set.append(example)

    print("Converting testing dataset to tf Examples...")
    print("Length testing X", len(teX))
    print("Length testing Y", len(teY))
    for i in range(len(teX)):
        example = _write_example(teX[i], np.where(teY[i] == 1)[0][0])
        test_set.append(example)

    print("Writing modified Examples into TFRecord file")
    # Rewrites all the files
    with tf.python_io.TFRecordWriter(train_file) as writer:
        for example in train_set:
            writer.write(example.SerializeToString())

    print("Writing testing Examples into TFRecord file")
    with tf.python_io.TFRecordWriter(test_file) as writer:
        for example in test_set:
            writer.write(example.SerializeToString())

    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--method',
        type=str,
        default='r',
        help='choose either r or cp where. r means resize to hxw size or cp means crop and pad to hxw'
    )
    parser.add_argument(
        '--h',
        type=int,
        default=32,
        help='height of resulting image'
    )
    parser.add_argument(
        '--w',
        type=int,
        default=32,
        help='width of resulting image'
    )
    parser.add_argument(
        '--sampling',
        type=str,
        default='smote',
        help='default is smote. can also try adasyn or none'
    )

    FLAGS, unparsed = parser.parse_known_args()

    print(FLAGS)
    if (FLAGS.method == 'cp' or FLAGS.method == 'r') and \
            (FLAGS.sampling == 'smote' or FLAGS.sampling == 'adasyn' or FLAGS.sampling == 'none'):
        tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    else:
        print('method is incorrect.')
