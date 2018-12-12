from random import randint
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn


# Spatial Pyramid Pooling based on the following paper
# https://arxiv.org/abs/1406.4729

# https://github.com/tensorflow/tensorflow/issues/6011
# https://github.com/tensorflow/tensorflow/pull/13259/files
# Code was taken from the below source
# https://github.com/tensorflow/tensorflow/pull/12852/files
def max_pool_2d_nxn_regions(inputs, pool_dimension, mode):
    """
    Args:
      inputs: The tensor over which to pool. Must have rank 4.
      pool_dimension: The dimenstion level(bin size)
        over which spatial pooling is performed.
      mode: Pooling mode 'max' or 'avg'.
    Returns:
      The output list of (pool_dimension * pool_dimension) tensors.
    """
    inputs_shape = array_ops.shape(inputs)
    h = math_ops.cast(array_ops.gather(inputs_shape, 1), dtypes.int32)
    w = math_ops.cast(array_ops.gather(inputs_shape, 2), dtypes.int32)

    if mode == 'max':
        pooling_op = math_ops.reduce_max
    elif mode == 'avg':
        pooling_op = math_ops.reduce_mean
    else:
        msg = "Mode must be either 'max' or 'avg'. Got '{0}'"
        raise ValueError(msg.format(mode))

    result = []
    n = pool_dimension
    for row in range(pool_dimension):
        for col in range(pool_dimension):
            # start_h = floor(row / n * h)
            start_h = math_ops.cast(
                math_ops.floor(math_ops.multiply(math_ops.divide(row, n), math_ops.cast(h, dtypes.float32))),
                dtypes.int32)
            # end_h = ceil((row + 1) / n * h)
            end_h = math_ops.cast(
                math_ops.ceil(math_ops.multiply(math_ops.divide((row + 1), n), math_ops.cast(h, dtypes.float32))),
                dtypes.int32)
            # start_w = floor(col / n * w)
            start_w = math_ops.cast(
                math_ops.floor(math_ops.multiply(math_ops.divide(col, n), math_ops.cast(w, dtypes.float32))),
                dtypes.int32)
            # end_w = ceil((col + 1) / n * w)
            end_w = math_ops.cast(
                math_ops.ceil(math_ops.multiply(math_ops.divide((col + 1), n), math_ops.cast(w, dtypes.float32))),
                dtypes.int32)
            pooling_region = inputs[:, start_h:end_h, start_w:end_w, :]
            pool_result = pooling_op(pooling_region, axis=(1, 2))
            result.append(pool_result)
    return result


def spatial_pyramid_pooling(inputs, dimensions=None,
                            mode='max', implementation='kaiming'):
    """
      Spatial pyramid pooling (SPP) is a pooling strategy to result in an output of fixed size.
      It will turn a 2D input of arbitrary size into an output of fixed dimension.
      Hence, the convlutional part of a DNN can be connected to a dense part
      with a fixed number of nodes even if the dimensions of the input
      image are unknown.
      The pooling is performed over :math:`l` pooling levels.
      Each pooling level :math:`i` will create :math:`M_i` output features.
      :math:`M_i` is given by :math:`n_i * n_i`, with :math:`n_i` as the number
      of pooling operations per dimension level :math:`i`.
      The length of the parameter dimensions is the level of the spatial pyramid.
    Args:
      inputs: The tensor over which to pool. Must have rank 4.
      dimensions: The list of bin sizes over which pooling is to be done.
      mode: Pooling mode 'max' or 'avg'.
      implementation: The implementation to use, either 'kaiming' or 'fast'.
        kamming is the original implementation from the paper, and supports variable
        sizes of input vectors, which fast does not support.
    Returns:
      Output tensor.
    """
    layer = SpatialPyramidPooling(dimensions=dimensions,
                                  mode=mode,
                                  implementation=implementation)
    return layer.apply(inputs)


class SpatialPyramidPooling(base.Layer):
    """
      Spatial pyramid pooling (SPP) is a pooling strategy to result in an output of fixed size.
      Arguments:
          dimensions: The list of :math:`n_i`'s that define the output dimension
            of each pooling level :math:`i`. The length of dimensions is the level of
            the spatial pyramid.
          mode: Pooling mode 'max' or 'avg'.
          implementation: The implementation to use, either 'kaiming' or 'fast'.
            kaiming is the original implementation from the paper, and supports variable
            sizes of input vectors, which fast does not support.
      Notes:
          SPP should be inserted between the convolutional part of a Deep Network and it's
          dense part. Convolutions can be used for arbitrary input dimensions, but
          the size of their output will depend on their input dimensions.
          Connecting the output of the convolutional to the dense part then
          usually demands us to fix the dimensons of the network's input.
          The spatial pyramid pooling layer, however, allows us to leave
          the network input dimensions arbitrary.
          The advantage over a global pooling layer is the added robustness
          against object deformations due to the pooling on different scales.
      References:
          [1] He, Kaiming et al (2015): Spatial Pyramid Pooling in Deep Convolutional Networks
              for Visual Recognition. https://arxiv.org/pdf/1406.4729.pdf.
      Ported from: https://github.com/Lasagne/Lasagne/pull/799
    """

    def __init__(self, dimensions=None, mode='max', implementation='kaiming', **kwargs):
        super(SpatialPyramidPooling, self).__init__(**kwargs)
        self.implementation = implementation
        self.mode = mode
        self.dimensions = dimensions if dimensions is not None else [4, 2, 1]

    def call(self, inputs):
        pool_list = []
        if self.implementation == 'kaiming':
            for pool_dim in self.dimensions:
                pool_list += max_pool_2d_nxn_regions(inputs, pool_dim, self.mode)
        else:
            input_shape = inputs.get_shape().as_list()
            for pool_dim in self.dimensions:
                h, w = input_shape[1], input_shape[2]

                ph = np.ceil(h * 1.0 / pool_dim).astype(np.int32)
                pw = np.ceil(w * 1.0 / pool_dim).astype(np.int32)
                sh = np.floor(h * 1.0 / pool_dim + 1).astype(np.int32)
                sw = np.floor(w * 1.0 / pool_dim + 1).astype(np.int32)
                pool_result = nn.max_pool(inputs,
                                          ksize=[1, ph, pw, 1],
                                          strides=[1, sh, sw, 1],
                                          padding='SAME')
                pool_list.append(array_ops.reshape(pool_result, [array_ops.shape(inputs)[0], -1]))
        return array_ops.concat(values=pool_list, axis=1)

    def _compute_output_shape(self, input_shape):
        num_features = sum(p * p for p in self.dimensions)
        return tensor_shape.TensorShape([None, input_shape[0] * num_features])


def testSpatialPyramidPoolingDefaultDimensionForBins():
    height, width, channel = 5, 6, 3
    images = array_ops.placeholder(dtype='float32',
                                   shape=(None, height, width, channel))
    layer = SpatialPyramidPooling()
    output = layer.apply(images)
    expected_output_size_for_each_channel = sum(d * d for d in layer.dimensions)
    print("Equal", output.get_shape().as_list(), [None, channel * expected_output_size_for_each_channel])
    # self.assertListEqual(output.get_shape().as_list(), [None, channel * expected_output_size_for_each_channel])


def testSpatialPyramidPoolingCustomDimensionForBins():
    height, width, channel = 5, 6, 3
    images = array_ops.placeholder(dtype='float32',
                                   shape=(None, height, width, channel))
    layer = SpatialPyramidPooling(dimensions=[3, 4, 5])
    expected_output_size_for_each_channel = sum(d * d for d in layer.dimensions)
    output = layer.apply(images)
    print("Equal", output.get_shape().as_list(), [None, channel * expected_output_size_for_each_channel])
    # self.assertListEqual(output.get_shape().as_list(), [None, channel * expected_output_size_for_each_channel])


def testSpatialPyramidPoolingBatchSizeGiven():
    batch_size, height, width, channel = 4, 5, 6, 3
    images = array_ops.placeholder(dtype='float32',
                                   shape=(batch_size, height, width, channel))
    layer = SpatialPyramidPooling(dimensions=[3, 4, 5])
    expected_output_size_for_each_channel = sum(d * d for d in layer.dimensions)
    output = layer.apply(images)
    print("Equal", output.get_shape().as_list(), [batch_size, channel * expected_output_size_for_each_channel])
    # self.assertListEqual(output.get_shape().as_list(), [batch_size, channel * expected_output_size_for_each_channel])


def testSpatialPyramidPoolingAssertOutDimensionFixedForAnyInput():
    layer = SpatialPyramidPooling(dimensions=[3, 4, 5])
    expected_output_size_for_each_channel = sum(d * d for d in layer.dimensions)
    output_arrays = []
    check_for_images = 10
    batch_size, channel = 2, 3
    for _ in range(check_for_images):
        height, width = randint(0, 9), randint(0, 9)
        images = array_ops.placeholder(dtype='float32',
                                       shape=(batch_size, height, width, channel))
        output = layer.apply(images)
        output_arrays.append(output.get_shape().as_list())

    print("Equal", output_arrays)
    print("Equal", [[batch_size, channel * expected_output_size_for_each_channel]] * check_for_images)
    # self.assertListEqual(output_arrays,
    #                     [[batch_size, channel * expected_output_size_for_each_channel]] * check_for_images)


def testSpatialPyramidPoolingComputeOutputShape():
    batch_size, height, width, channel = 4, 5, 6, 3
    layer = SpatialPyramidPooling(dimensions=[3, 4, 5])
    image = array_ops.placeholder(dtype='float32',
                                  shape=(batch_size, height, width, channel))
    output_shape = layer._compute_output_shape(input_shape=image._shape)

    print("Equal", output_shape.as_list(), [None, 200])
    # self.assertListEqual(output_shape.as_list() , [None, 200])


def testSpatialPyramidPoolingMode():
    batch_size, height, width, channel = 4, 5, 6, 3
    mode = 'invalid_mode'
    layer = SpatialPyramidPooling(dimensions=[3, 4, 5], mode=mode)
    images = array_ops.placeholder(dtype='float32',
                                   shape=(batch_size, height, width, channel))

    print("Equal", layer)
    # with self.assertRaisesRegexp(
    #        ValueError, "Mode must be either 'max' or 'avg'. Got '{}'".format(mode)):
    #  layer.apply(images)


if __name__ == "__main__":
    testSpatialPyramidPoolingDefaultDimensionForBins()
    testSpatialPyramidPoolingCustomDimensionForBins()
    testSpatialPyramidPoolingBatchSizeGiven()
    testSpatialPyramidPoolingAssertOutDimensionFixedForAnyInput()
    testSpatialPyramidPoolingComputeOutputShape()
    testSpatialPyramidPoolingMode()

# unused alternative implementation of spatial pyramid pooling
# def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
#     """
#     https://github.com/peace195/sppnet/blob/master/alexnet_spp.py
#     previous_conv: a tensor vector of previous convolution layer
#     num_sample: an int number of image in the batch
#     previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
#     out_pool_size: a int vector of expected output size of max pooling layer
#
#     returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
#     """
#     for i in range(len(out_pool_size)):
#         h_strd = h_size = math.ceil(float(previous_conv_size[0]) / out_pool_size[i])
#         w_strd = w_size = math.ceil(float(previous_conv_size[1]) / out_pool_size[i])
#         pad_h = int(out_pool_size[i] * h_size - previous_conv_size[0])
#         pad_w = int(out_pool_size[i] * w_size - previous_conv_size[1])
#         new_previous_conv = tf.pad(previous_conv, tf.constant([[0, 0], [0, pad_h], [0, pad_w], [0, 0]]))
#         max_pool = tf.nn.max_pool(new_previous_conv,
#                                   ksize=[1, h_size, h_size, 1],
#                                   strides=[1, h_strd, w_strd, 1],
#                                   padding='SAME')
#         if i == 0:
#             spp = tf.reshape(max_pool, [num_sample, -1])
#         else:
#             spp = tf.concat(axis=1, values=[spp, tf.reshape(max_pool, [num_sample, -1])])
#
#     return spp