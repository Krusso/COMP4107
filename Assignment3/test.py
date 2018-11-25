import sklearn.cluster as sk
from sklearn.utils import shuffle
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from sklearn.model_selection import KFold
import os

print('hello')
ones = np.ones((10,10))
print(ones)
with tf.Session() as sess:
    # print(ones)
    # layer = tf.Variable(ones)
    dropout = tf.nn.dropout(ones, keep_prob=0.5)
    print(sess.run(dropout))