import tensorflow as tf
import sys
import os
import argparse
from scipy import misc
import numpy as np
from urllib.request import urlretrieve
import tarfile
# import cv2
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
FLAGS=None

# https://www.tensorflow.org/tutorials/images/deep_cnn
# https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10/
DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
DATA_DIR = './tmp'

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 32

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def cifar10(path=None):
    # https://mattpetersen.github.io/load-cifar10-with-numpy
    r"""Return (train_images, train_labels, test_images, test_labels).

    Args:
        path (str): Directory containing CIFAR-10. Default is
            /home/USER/data/cifar10 or C:\Users\USER\data\cifar10.
            Create if nonexistant. Download CIFAR-10 if missing.

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels), each
            a matrix. Rows are examples. Columns of images are pixel values,
            with the order (red -> blue -> green). Columns of labels are a
            onehot encoding of the correct class.
    """
    url = 'https://www.cs.toronto.edu/~kriz/'
    tar = 'cifar-10-binary.tar.gz'
    files = ['cifar-10-batches-bin/data_batch_1.bin',
             'cifar-10-batches-bin/data_batch_2.bin',
             'cifar-10-batches-bin/data_batch_3.bin',
             'cifar-10-batches-bin/data_batch_4.bin',
             'cifar-10-batches-bin/data_batch_5.bin',
             'cifar-10-batches-bin/test_batch.bin']

    if path is None:
        # Set path to /home/USER/data/mnist or C:\Users\USER\data\mnist
        path = os.path.join(os.path.expanduser('~'), 'data', 'cifar10')

    # Create path if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Download tarfile if missing
    if tar not in os.listdir(path):
        urlretrieve(''.join((url, tar)), os.path.join(path, tar))
        print("Downloaded %s to %s" % (tar, path))

    # Load data from tarfile
    with tarfile.open(os.path.join(path, tar)) as tar_object:
        # Each file contains 10,000 color images and 10,000 labels
        fsize = 10000 * (32 * 32 * 3) + 10000

        # There are 6 files (5 train and 1 test)
        buffr = np.zeros(fsize * 6, dtype='uint8')

        # Get members of tar corresponding to data files
        # -- The tar contains README's and other extraneous stuff
        members = [file for file in tar_object if file.name in files]

        # Sort those members by name
        # -- Ensures we load train data in the proper order
        # -- Ensures that test data is the last file in the list
        members.sort(key=lambda member: member.name)

        # Extract data from members
        for i, member in enumerate(members):
            # Get member as a file object
            f = tar_object.extractfile(member)
            # Read bytes from that file object into buffr
            buffr[i * fsize:(i + 1) * fsize] = np.frombuffer(f.read(), 'B')

    # Parse data from buffer
    # -- Examples are in chunks of 3,073 bytes
    # -- First byte of each chunk is the label
    # -- Next 32 * 32 * 3 = 3,072 bytes are its corresponding image

    # Labels are the first byte of every chunk
    labels = buffr[::3073]

    # Pixels are everything remaining after we delete the labels
    pixels = np.delete(buffr, np.arange(0, buffr.size, 3073))
    images = pixels.reshape(-1, 32, 32, 3).astype('float32') / 255

    # Split into train and test
    train_images, test_images = images[:50000], images[50000:]
    train_labels, test_labels = labels[:50000], labels[50000:]

    def _onehot(integer_labels):
        """Return matrix whose rows are onehot encodings of integers."""
        n_rows = len(integer_labels)
        n_cols = integer_labels.max() + 1
        onehot = np.zeros((n_rows, n_cols), dtype='uint8')
        onehot[np.arange(n_rows), integer_labels] = 1
        return onehot

    return train_images, _onehot(train_labels), \
        test_images, _onehot(test_labels)




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
        features = tf.train.Features(
            feature={
                'image_raw' : _bytes_feature(image.tostring()),
                'label' : _int64_feature(label)
                }))
    return example

def modify_image(trX, trY):
    #Distort the images
    # with tf.Session() as sess:
        #for each image, we add one distortion
    distorted = []
    labels = []
    from scipy import ndimage
    sess = tf.InteractiveSession()
    #for each image we rotate 90 degrees
    for i in range(len(trX)):
        d_img = np.reshape(trX[i], )
        d_img = np.rot90(trX[i], 2, axes=(1,2))

        print(np.shape(d_img))
        distorted.append(d_img)
        from scipy.misc import toimage
        # img = plt.imshow(toimage(sess.run( tf.transpose(tf.reshape(d_img, shape=[3,32,32]), perm=[1,2,0] )    ) ), interpolation='nearest')
        img = plt.imshow(toimage(d_img.reshape(3, 32, 32)), interpolation='nearest')
        plt.show()
        img = plt.imshow(toimage(trX[i].reshape(3, 32, 32)), interpolation='nearest')
        plt.show()
    distorted.extend(trX)
    labels.extend(trY)
    labels.extend(trY)
    return shuffle(distorted, labels)

DISTORTED_IMAGE_DIR = ".\distorted_images"
if not os.path.exists(DISTORTED_IMAGE_DIR):
    os.makedirs(DISTORTED_IMAGE_DIR)

train_file = os.path.join(DISTORTED_IMAGE_DIR, 'train_set.tfrecords')
test_file = os.path.join(DISTORTED_IMAGE_DIR, 'test_set.tfrecords')

trX, trY, teX, teY = cifar10(path='./tmp')
print("Read CIFAR10 dataset")

modified_images, labels = modify_image(trX, trY)
print("Completed Modification of Images")

n = len(modified_images)
train_set = []
test_set = []
print("Converting modified dataset to tf Examples...")
for i in range(n):
    # print(labels[i])
    # print(np.where(labels[i]==1)[0][0])
    #convert to tf examples, labels one-hot encoding converted to integer (later parsing will need to convert back)
    example = _write_example(modified_images[i], np.where(labels[i]==1)[0][0]) 
    train_set.append(example)

print("Converting testing dataset to tf Examples...")
for i in range(len(teX)):
    example = _write_example(teX[i], np.where(teY[i]==1)[0][0]) 
    train_set.append(example)

print("Writing modified Examples into TFRecord file")
#Rewrites all the files
with tf.python_io.TFRecordWriter(train_file) as writer: 
    for example in train_set:
        writer.write(example.SerializeToString())

print("Writing testing Examples into TFRecord file")
with tf.python_io.TFRecordWriter(test_file) as writer: 
    for example in train_set:
        writer.write(example.SerializeToString())

print("done")

        


