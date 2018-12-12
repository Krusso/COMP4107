import tensorflow as tf
import sys
import os
import argparse
from scipy import misc
import numpy as np
from urllib.request import urlretrieve
import tarfile
import cv2
import glob
from sklearn.utils import shuffle
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import imblearn.over_sampling as imb

FLAGS = None


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
            im = np.multiply(im, 1.0/255.0)
            h, w, c = im.shape
            # cv2.imshow('original',im)
            # print(im.shape)
            # print(height, width)
            if FLAGS.method == 'cp':
                # im = tf.image.resize_image_with_crop_or_pad(im, [height, width])
                # im = cv2.resize(im, [height, width], interpolation=cv2.INTER_AREA)
                if w > width: #if w is greater than resize width, crop it about the center.
                    im = im[:, w//2 - width//2: w//2 + width//2]
                else: #if w is smaller than the resize width, pad it with 0 about the center
                    num_padding = width - w
                    left_pad = num_padding//2
                    right_pad = num_padding - left_pad
                    im = cv2.copyMakeBorder(im, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)

                # print(im)
                if h > height:
                    im = im[h//2 - height//2: h//2 + height//2]
                else: 
                    num_padding = height - h
                    top_pad = num_padding//2
                    bot_pad = num_padding - top_pad
                    im = cv2.copyMakeBorder(im, top_pad, bot_pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
                # print(im.shape)
            elif FLAGS.method == 'r':
                im = cv2.resize(im, (height, width), interpolation=cv2.INTER_AREA)
                
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
    dataset = dataset.flatten().reshape(6899,height*width*3)
    x = [[1, 2],
         [1, 2],
         [3, 4],
         [7, 4],
         [3, 6],
         [4, 5],
         [6, 2]]
    
    y = [1, 1, 2, 2, 2, 2, 1]
    print('Fitting samples...')
    if FLAGS.sampling == 'smote':
        dataset, labels = imb.SMOTE(n_jobs=4).fit_resample(dataset, labels)
    elif FLAGS.sampling == 'adasyn':
        dataset, labels = imb.ADASYN(sampling_strategy='all', n_jobs=4).fit_resample(x, y)
        # dataset, labels = sm.fit_resample(dataset, labels)

    dataset = dataset.reshape(dataset.shape[0], height, width, 3)

    numAirplane = 0
    nextOne = 0
    for l in labels:
        if l[0] == 1:
            numAirplane += 1
        elif l[1] == 1:
            nextOne += 1
    print(numAirplane)
    print(nextOne)
    print(len(labels))
    print(len(dataset))
    print("Finished generating new data")

    # trainData, trainLabel, testData, testLabel
    # TODO: @Michael/@Krystian shuffle the data before returning it and set the bounds for training/testing correctly
    
    #num examples
    num_examples = dataset.shape[0]
    kfold = 10
    split_index = num_examples//kfold

    return dataset[:split_index], labels[:split_index], \
        dataset[split_index:], labels[split_index:]


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


def modify_image(trX, trY):
    """
        Distort each image by randomly rotating 90 degrees and adding pepper&salt to the image
        input: image and labels
        output: shuffled modified images + original images, labels
    """
    distorted = []
    labels = []
    for i in range(len(trX)):
        d_img = salt_pepper(np.rot90(trX[i], np.random.randint(1,4)))
        distorted.append(d_img)
        
        # cv2.imshow('distorted', d_img)
        # cv2.waitKey(0)

    distorted.extend(trX)
    labels.extend(trY)
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



    train_file = os.path.join(DISTORTED_IMAGE_DIR, 'train_set_h{}w{}_{}_{}.tfrecords'.format(FLAGS.h,FLAGS.w, FLAGS.method, FLAGS.sampling))
    test_file = os.path.join(DISTORTED_IMAGE_DIR, 'test_set_h{}w{}_{}_{}.tfrecords'.format(FLAGS.h,FLAGS.w, FLAGS.method, FLAGS.sampling))
    

    # TODO: @Michael get the data augmentation call back into the pipeline
    trX, trY = modify_image(trX, trY)
    print("Completed Modification of Images")

    

    n = len(trX)
    train_set = []
    test_set = []
    print("Converting modified dataset to tf Examples...")
    for i in range(n):
        # convert to tf examples, labels one-hot encoding converted to integer (later parsing will need to convert back)
        example = _write_example(trX[i], np.where(trY[i] == 1)[0][0])
        train_set.append(example)

    print("Converting testing dataset to tf Examples...")
    for i in range(len(teX)):
        example = _write_example(teX[i], np.where(teY[i] == 1)[0][0])
        train_set.append(example)

    print("Writing modified Examples into TFRecord file")
    # Rewrites all the files
    with tf.python_io.TFRecordWriter(train_file) as writer:
        for example in train_set:
            writer.write(example.SerializeToString())

    print("Writing testing Examples into TFRecord file")
    with tf.python_io.TFRecordWriter(test_file) as writer:
        for example in train_set:
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
        help='default is smote. can also try adasyn'
    )    





    FLAGS, unparsed = parser.parse_known_args()

    from collections import Counter
    from sklearn.datasets import make_classification
    from imblearn.over_sampling import ADASYN # doctest: +
    
    x, y = make_classification(n_classes=2, class_sep=2, \
    weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0, \
     n_features=20, n_clusters_per_class=1, n_samples=1000, \
     random_state=10)
    
    
    print(x.shape)
    print(y.shape)
    # print(y[1])

    print('Fitting samples...')
    
    x = []
    y = []
    for i in range(25):
        x.append([np.random.random_integers(0, 10), np.random.random_integers(0,10)])
        y.append(np.random.random_integers(0,4))
    x = np.asarray(x)
    
    y = np.asarray(y)
    print(x.shape)
    print(y.shape)

    print(x)
    print(y)

    dataset, labels = imb.ADASYN().fit_resample(x, y)
    print(dataset, labels)

    print(x.shape)
    print(y.shape)
    # print(FLAGS)
    # if (FLAGS.method == 'cp' or FLAGS.method == 'r') and (FLAGS.sampling == 'smote' or FLAGS.sampling == 'adasyn'):
    #     tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    # else:
    #     print('method is incorrect.')