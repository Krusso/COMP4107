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

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
            h, w, c = im.shape
            # cv2.imshow('original',im)
            # print(im.shape)
            # print(height, width)
            if cropAndPad:
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
            else:
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
    # import imblearn.over_sampling as imb
    # sm = imb.SMOTE(k_neighbors=1)
    # sm = imb.ANADYS...
    # x = [[[1, 2, 3]],
    #      [[3, 4, 5, 6]],
    #      [[1, 2]]]
    # x = [[1, 2],
    #      [1, 2],
    #      [3, 4],
    #      [7, 4],
    #      [3, 6],
    #      [4, 5]]
    #
    # y = [1, 1, 2, 2, 2, 2]
    #
    # X_train_res, y_train_res = sm.fit_sample(x, y)
    # print(X_train_res)
    # print(y_train_res)

    print("Finished generating new data")

    # trainData, trainLabel, testData, testLabel
    # TODO: @Michael/@Krystian shuffle the data before returning it and set the bounds for training/testing correctly
    return dataset[0:400], labels[0:400], \
        dataset[400:600], labels[400:600]


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
    amount = 0.02
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
    # Distort the images
    # with tf.Session() as sess:
    # for each image, we add one distortion
    distorted = []
    labels = []
    # for each image we rotate 90 degrees
    for i in range(len(trX)):
        # if i % 1000 == 0:
        print("on image", i)

        # d_img = sess.run(tf.image.random_flip_left_right(
        #                trX[i].transpose(1, 2, 0))).transpose(2, 0, 1)
        d_img = trX[i]

        d_img = salt_pepper(np.rot90(d_img.transpose(1, 2, 0), np.random.randint(0, 4))).transpose(2, 0, 1)

        distorted.append(d_img)

        # print(d_img.shape)
        # img = plt.imshow(d_img.transpose(1, 2, 0))
        # plt.show()
        # img = plt.imshow(trX[i].transpose(1, 2, 0))
        # plt.show()
        # break
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

    trX, trY, teX, teY = natural_images(path='./natural_images', height=FLAGS.h, width=FLAGS.w, cropAndPad=cropAndPad, short=True)
    # Different sizes we can try for the height/width of the modified images
    # trX, trY, teX, teY = cifar10(path='./natural_images', height=32, width=32, cropAndPad=True)
    # trX, trY, teX, teY = cifar10(path='./natural_images', height=64, width=64, cropAndPad=False)
    # trX, trY, teX, teY = cifar10(path='./natural_images', height=64, width=64, cropAndPad=True)
    # trX, trY, teX, teY = cifar10(path='./natural_images', height=64, width=128, cropAndPad=False)
    # trX, trY, teX, teY = cifar10(path='./natural_images', height=32, width=128, cropAndPad=True)



    train_file = os.path.join(DISTORTED_IMAGE_DIR, 'train_set_h{}w{}_{}.tfrecords'.format(FLAGS.h,FLAGS.w, FLAGS.method))
    test_file = os.path.join(DISTORTED_IMAGE_DIR, 'test_set_h{}w{}_{}.tfrecords'.format(FLAGS.h,FLAGS.w, FLAGS.method))
    print("Read natural images dataset")

    # TODO: @Michael get the data augmentation call back into the pipeline
    # modified_images, labels = modify_image(trX, trY)
    # print("Completed Modification of Images")

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
        '--type',
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


    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    if FLAGS.method != 'cp' or FLAGS.method != 'r':
        print('method is incorrect.')
    else:
        tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)