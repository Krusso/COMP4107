import math
import tensorflow as tf
import imblearn.over_sampling as imb
import os
import numpy as np


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


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def create_model(X, p_keep_conv, p_keep_hidden, spatial=False):
    print("Creating model")
    w_1 = init_weights([3, 3, 3, 32])  # 3x3x3 conv, 32 outputs
    tf.summary.histogram("weights of first convolution layer 3x3x3x32 model5", w_1)

    w_2 = init_weights([3, 3, 32, 32])  # 3x3x3 conv, 32 outputs
    tf.summary.histogram("weights of second convolution layer 3x3x32x32 model5", w_2)

    w_3 = init_weights([3, 3, 32, 32])  # 3x3x3 conv, 32 outputs
    tf.summary.histogram("weights of third convolution layer 3x3x32x32 model5", w_3)

    w_fc = init_weights([32 * 8 * 8, 625])  # FC 32 * 14 * 14 inputs, 625 outputs
    tf.summary.histogram("weights of fully connected 625 neuron first layer model5", w_fc)
    w_o = init_weights([625, 8])  # FC 625 inputs, 8 outputs (labels)
    tf.summary.histogram("weights of 8 neuron output layer model5", w_o)

    l1a = tf.nn.relu(tf.nn.conv2d(X, w_1,  # X=(128,32,32,3) l1a shape=(?, 32, 32, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))

    l2a = tf.nn.relu(tf.nn.conv2d(l1a, w_2,  # l1a shape=(?, 32, 32, 32) l2a shape=(?, 32, 32, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))

    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],  # l1 shape=(?, 8, 8, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w_3,
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    if spatial:
        # do spatial pooling
        print(l3.get_shape())
        l3 = spatial_pyramid_pool(l3,
                                  int(l3.get_shape()[1]),
                                  [int(l3.get_shape()[2]), int(l3.get_shape()[3])],
                                  [8, 6, 4])

    l3 = tf.reshape(l3, [-1, w_fc.get_shape().as_list()[0]])  # reshape to (?, 14x14x32)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w_fc))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)

    return pyx


# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
# confusion matrix


def parse_example(example):
    features = {'image_raw': tf.FixedLenFeature((), tf.string, default_value=""),
                'label': tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(example, features)

    image = tf.decode_raw(parsed_features['image_raw'], tf.float32)
    image = tf.reshape(image, [32, 32, 3])

    label = tf.cast(parsed_features['label'], tf.int64)
    return image, tf.one_hot(label, 8)


def input_fn(filenames, shuffle_buff=100, batch_size=128):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.shuffle(shuffle_buff)
    dataset = dataset.map(lambda example: parse_example(example))
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    return dataset


batch_size = 32
height=32
width=32

train_set = input_fn(filenames=["distorted_images/train_set.tfrecords"],
                     batch_size=batch_size)

test_set = input_fn(filenames=["distorted_images/test_set.tfrecords"],
                    batch_size=batch_size)
# Training iterators
train_iterator = train_set.make_initializable_iterator()
test_iterator = test_set.make_initializable_iterator()

# Pipeline for feeding into the net, gets the next batch in dataset
next_train_batch = train_iterator.get_next()
next_test_batch = test_iterator.get_next()

# Used to reset the iterator
train_init_op = train_iterator.initializer
test_init_op = test_iterator.initializer

X = tf.placeholder("float", [None, 32, 32, 3], name='image')
Y = tf.placeholder("float", [None, 8], name='label')
batch_accuracies = tf.placeholder("float", [None])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

attempt = 0

while True:
    if os.path.isdir('./logs/attempt_{}'.format(attempt)):
        attempt += 1
    else:
        break

for name, model in list([("Model", create_model(X, p_keep_conv, p_keep_hidden, spatial=False))]):
    print("Starting model", name)
    py_x = model

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(py_x, 1)

    mean_accuracy = tf.reduce_mean(tf.cast(batch_accuracies, tf.float32))

    # Launch the graph in a session
    with tf.Session() as sess:
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()

        testing_accuracy = tf.summary.scalar('accuracy', mean_accuracy)
        merged_testing_accuracy = tf.summary.merge([testing_accuracy])

        result_dir = './logs/attempt_{}/{}'.format(attempt, name)
        summary_writer = tf.summary.FileWriter(result_dir, graph=sess.graph)

        # you need to initialize all variables
        tf.global_variables_initializer().run()

        for epoch in range(15):
            # Initialize the training iterator to consume training data
            sess.run(train_init_op)
            while True:
                # as long as the iterator has not hit the end, continue to consume training data
                try:
                    images, labels = sess.run(next_train_batch)
                    # print(images)
                    train_summary, _ = sess.run([merged, train_op], feed_dict={X: images,
                                                                               Y: labels,
                                                                               p_keep_conv: 0.8, p_keep_hidden: 0.5})
                except tf.errors.OutOfRangeError:
                    # end of training epoch
                    summary_writer.add_summary(train_summary, epoch)
                    break

            # Initialize the validation iterator to consume validation data
            sess.run(test_init_op)
            total_accuracy = []
            while True:
                try:
                    images, labels = sess.run(next_test_batch)
                    test_batch_accuracy = np.mean(np.argmax(labels, axis=1) ==
                                                  sess.run(predict_op, feed_dict={X: images,
                                                                                  Y: labels,
                                                                                  p_keep_conv: 1.0,
                                                                                  p_keep_hidden: 1.0}))
                    total_accuracy.append(test_batch_accuracy)
                except tf.errors.OutOfRangeError:
                    test_accuracy_summary, m_accuracy = sess.run([merged_testing_accuracy, mean_accuracy],
                                                                 feed_dict={batch_accuracies: total_accuracy})

                    summary_writer.add_summary(test_accuracy_summary, epoch)
                    # end of validation
                    print("Epoch {}: testing accuracy: {}".format(epoch, m_accuracy))

                    if (epoch + 1) % 5 == 0:
                        save_path = './logs/attempt_{}/model_checkpoint/model_{}_epoch{}.ckpt'.format(attempt, name,
                                                                                                      epoch + 1)
                        saver.save(sess, save_path)
                        print("Saved model at {}".format(save_path))
                    break

            summary_writer.flush()
