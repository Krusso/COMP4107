import math
import tensorflow as tf

# https://github.com/tensorflow/tensorflow/issues/6720
# https://hackernoon.com/how-tensorflows-tf-image-resize-stole-60-days-of-my-life-aba5eb093f35

# https://github.com/tensorflow/tensorflow/issues/6011
# https://github.com/tensorflow/tensorflow/pull/13259/files
# https://github.com/tensorflow/tensorflow/pull/12852/files


# Spatial Pyramid Pooling block
# https://arxiv.org/abs/1406.4729
def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    """
    https://github.com/peace195/sppnet/blob/master/alexnet_spp.py
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer

    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    """
    for i in range(len(out_pool_size)):
        h_strd = h_size = math.ceil(float(previous_conv_size[0]) / out_pool_size[i])
        w_strd = w_size = math.ceil(float(previous_conv_size[1]) / out_pool_size[i])
        pad_h = int(out_pool_size[i] * h_size - previous_conv_size[0])
        pad_w = int(out_pool_size[i] * w_size - previous_conv_size[1])
        new_previous_conv = tf.pad(previous_conv, tf.constant([[0, 0], [0, pad_h], [0, pad_w], [0, 0]]))
        max_pool = tf.nn.max_pool(new_previous_conv,
                                  ksize=[1, h_size, h_size, 1],
                                  strides=[1, h_strd, w_strd, 1],
                                  padding='SAME')
        if i == 0:
            spp = tf.reshape(max_pool, [num_sample, -1])
        else:
            spp = tf.concat(axis=1, values=[spp, tf.reshape(max_pool, [num_sample, -1])])

    return spp
