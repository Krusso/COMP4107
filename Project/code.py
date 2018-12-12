import tensorflow as tf

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import argparse
from SpatialPyramidPooling import SpatialPyramidPooling as spp


def init_weights(shape):
    print("Creating shapes size", shape)
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def create_model(X, p_keep_conv, p_keep_hidden, height, width, spatial=False):
    print("Creating model")
    w_1 = init_weights([3, 3, 3, 32])  # 3x3x3 conv, 32 outputs
    tf.summary.histogram("weights of first convolution layer 3x3x3x32 model", w_1)

    w_2 = init_weights([3, 3, 32, 32])  # 3x3x3 conv, 32 outputs
    tf.summary.histogram("weights of second convolution layer 3x3x32x32 model", w_2)

    w_3 = init_weights([3, 3, 32, 32])  # 3x3x3 conv, 32 outputs
    tf.summary.histogram("weights of third convolution layer 3x3x32x32 model", w_3)

    if spatial:
        w_fc = init_weights([1600, 625])  # FC 3^2 + 4^2 + 5^2 + 6^2 inputs, 625 outputs
    else:
        w_fc = init_weights([int(32 * (height / 8) * (width / 8)), 625])  # FC 32 * 8 * 8 inputs, 625 outputs

    tf.summary.histogram("weights of fully connected 625 neuron first layer model5", w_fc)
    w_o = init_weights([625, 8])  # FC 625 inputs, 8 outputs (labels)
    tf.summary.histogram("weights of 8 neuron output layer model5", w_o)

    l1a = tf.nn.relu(tf.nn.conv2d(X, w_1,  # X=(batchSize, height, width, 3) l1a shape=(?, height, width, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))

    l1a = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],  # l1 shape=(?, height/2, width/2, 32)
                         strides=[1, 2, 2, 1], padding='SAME')

    l2a = tf.nn.relu(tf.nn.conv2d(l1a, w_2,  # l2a shape=(?, height/2, width/2, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))

    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],  # l2 shape=(?, height/4, width/4, 32)
                        strides=[1, 2, 2, 1], padding='SAME')

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w_3,  # l2a shape=(?, height/4, width/4, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))

    # either do spatial pooling or regular max pooling
    if spatial:
        # spatial pooling
        layer = spp(dimensions=[3, 4, 5])  # results in (3^2 + 4^2 + 5^2) * feature outputs
        l3 = layer.apply(l3a)
    else:
        l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],  # l2a shape=(?, height/8, width/8, 32)
                            strides=[1, 2, 2, 1], padding='SAME')

    l3 = tf.reshape(l3, [-1, w_fc.get_shape().as_list()[0]])  # reshape to (?, 32 * (height / 8) * (width / 8))
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w_fc))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)

    return pyx


def parse_example(example, height, width):
    features = {'image_raw': tf.FixedLenFeature((), tf.string, default_value=""),
                'label': tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(example, features)

    image = tf.decode_raw(parsed_features['image_raw'], tf.float32)
    image = tf.reshape(image, [height, width, 3])

    label = tf.cast(parsed_features['label'], tf.int64)
    return image, tf.one_hot(label, 8)


def input_fn(filenames, height, width, shuffle_buff=100, batch_size=128):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.shuffle(shuffle_buff)
    dataset = dataset.map(lambda example: parse_example(example, height, width))
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    return dataset


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


parser = argparse.ArgumentParser()

parser.add_argument(
    '--spatial',
    type=bool,
    default=False,
    help='either True or False'
)

parser.add_argument(
    '--verbose',
    type=bool,
    default=False,
    help='either True or False'
)

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

batch_size = 32

train_set = input_fn(
    height=FLAGS.h,
    width=FLAGS.w,
    filenames=[
        'distorted_images/train_set_h{}w{}_{}_{}.tfrecords'.format(FLAGS.h, FLAGS.w, FLAGS.method, FLAGS.sampling)],
    batch_size=batch_size)

test_set = input_fn(
    height=FLAGS.h,
    width=FLAGS.w,
    filenames=[
        'distorted_images/test_set_h{}w{}_{}_{}.tfrecords'.format(FLAGS.h, FLAGS.w, FLAGS.method, FLAGS.sampling)],
    batch_size=32)

# Training iterators
train_iterator = train_set.make_initializable_iterator()
test_iterator = test_set.make_initializable_iterator()

# Pipeline for feeding into the net, gets the next batch in dataset
next_train_batch = train_iterator.get_next()
next_test_batch = test_iterator.get_next()

# Used to reset the iterator
train_init_op = train_iterator.initializer
test_init_op = test_iterator.initializer

X = tf.placeholder("float", [None, FLAGS.h, FLAGS.w, 3], name='image')
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

for name, model in list(
        [("Model", create_model(X, p_keep_conv, p_keep_hidden, FLAGS.h, FLAGS.w, spatial=FLAGS.spatial))]):
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
            y_tests = []
            y_preds = []
            while True:
                try:
                    images, labels = sess.run(next_test_batch)

                    y_pred = sess.run(predict_op, feed_dict={X: images,
                                                             Y: labels,
                                                             p_keep_conv: 1.0,
                                                             p_keep_hidden: 1.0})

                    y_tests.extend(np.argmax(labels, axis=1))
                    y_preds.extend(y_pred)

                    test_batch_accuracy = np.mean(np.argmax(labels, axis=1) == y_pred)
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

                    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
                    # confusion matrix

                    plt.figure()

                    cnf_matrix = confusion_matrix(y_tests, y_preds)
                    print("confusion matrix", cnf_matrix)
                    if FLAGS.verbose:
                        plot_confusion_matrix(cnf_matrix, classes=["airplane",
                                                                   "car",
                                                                   "cat",
                                                                   "dog",
                                                                   "flower",
                                                                   "fruit",
                                                                   "motorbike",
                                                                   "person"],
                                              title='Confusion matrix epoch {}'.format(epoch))
                        plt.show()
                    break

            summary_writer.flush()
