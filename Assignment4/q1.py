import tensorflow as tf
import numpy as np
import os
import tarfile
from urllib.request import urlretrieve

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


batch_size = 128


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model1(X, p_keep_conv, p_keep_hidden):
    w = init_weights([3, 3, 3, 32])  # 3x3x3 conv, 32 outputs
    tf.summary.histogram("weights of first convolution layer 3x3x3x32", w)
    w_fc = init_weights([32 * 16 * 16, 625])  # FC 32 * 14 * 14 inputs, 625 outputs
    tf.summary.histogram("weights of fully connected 625 neuron first layer", w_fc)
    w_o = init_weights([625, 10])  # FC 625 inputs, 10 outputs (labels)
    tf.summary.histogram("weights of 10 neuron output layer", w_o)

    l1a = tf.nn.relu(tf.nn.conv2d(X, w,  # l1a shape=(?, 32, 32, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],  # l1 shape=(?, 16, 16, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l3 = tf.reshape(l1, [-1, w_fc.get_shape().as_list()[0]])  # reshape to (?, 14x14x32)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w_fc))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx


def model2(X, p_keep_conv, p_keep_hidden):
    w = init_weights([3, 3, 3, 32])  # 3x3x3 conv, 32 outputs
    tf.summary.histogram("weights of first convolution layer 3x3x3x32", w)
    w_1 = init_weights([3, 3, 32, 32])  # 3x3x32 conv, 32 outputs
    tf.summary.histogram("weights of second convolution layer 3x3x32x32", w_1)
    w_fc = init_weights([32 * 8 * 8, 625])  # FC 32 * 14 * 14 inputs, 625 outputs
    tf.summary.histogram("weights of fully connected 625 neuron first layer", w_fc)
    w_o = init_weights([625, 10])  # FC 625 inputs, 10 outputs (labels)
    tf.summary.histogram("weights of 10 neuron output layer", w_o)

    l1a = tf.nn.relu(tf.nn.conv2d(X, w,  # l1a shape=(?, 32, 32, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],  # l1 shape=(?, 16, 16, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w_1,  # l2a shape=(?, 16, 16, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],  # l2 shape=(?, 8, 8, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3 = tf.reshape(l2, [-1, w_fc.get_shape().as_list()[0]])  # reshape to (?, 14x14x32)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w_fc))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx


def model3(X, p_keep_conv, p_keep_hidden):
    w = init_weights([3, 3, 3, 32])  # 3x3x3 conv, 32 outputs
    tf.summary.histogram("weights of first convolution layer 3x3x3x32", w)
    w_1 = init_weights([3, 3, 32, 32])  # 3x3x3 conv, 32 outputs
    tf.summary.histogram("weights of second convolution layer 3x3x32x32", w_1)
    w_fc = init_weights([32 * 8 * 8, 625])  # FC 32 * 14 * 14 inputs, 625 outputs
    tf.summary.histogram("weights of fully connected 625 neuron first layer", w_fc)
    w_o = init_weights([625, 10])  # FC 625 inputs, 10 outputs (labels)
    tf.summary.histogram("weights of 10 neuron output layer", w_o)

    l1a = tf.nn.relu(tf.nn.conv2d(X, w,  # l1a shape=(?, 32, 32, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))

    tf.summary.image(l1a)

    l1a = tf.nn.relu(tf.nn.conv2d(l1a, w_1,  # l1a shape=(?, 32, 32, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 4, 4, 1],  # l1 shape=(?, 8, 8, 32)
                        strides=[1, 4, 4, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l3 = tf.reshape(l1, [-1, w_fc.get_shape().as_list()[0]])  # reshape to (?, 14x14x32)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w_fc))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx


def model4(X, p_keep_conv, p_keep_hidden):
    return None


def model5(X, p_keep_conv, p_keep_hidden):
    return None


trX, trY, teX, teY = cifar10(path='./tmp')

X = tf.placeholder("float", [None, 32, 32, 3])
Y = tf.placeholder("float", [None, 10])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

attempt = 0

while True:
    if os.path.isdir('./logs/attempt_{}'.format(attempt)):
        attempt += 1
    else:
        break

for name, py_x in list(("model 3", [model3(X, p_keep_conv, p_keep_hidden)])):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(py_x, 1)

    batch_accuracies = tf.placeholder("float", [None])
    mean_accuracy = tf.reduce_mean(tf.cast(batch_accuracies, tf.float32))
    tf.summary.scalar('accuracy', mean_accuracy)

    # Launch the graph in a session
    with tf.Session() as sess:
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        result_dir = './logs/attempt_{}/model_{}'.format(attempt, name)
        summary_writer = tf.summary.FileWriter(result_dir, graph=sess.graph)
        # you need to initialize all variables
        tf.global_variables_initializer().run()

        for i in range(15):
            training_batch = zip(range(0, len(trX), batch_size),
                                 range(batch_size, len(trX) + 1, batch_size))

            for start, end in training_batch:
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                              p_keep_conv: 0.8, p_keep_hidden: 0.5})

            saver.save(sess, 'cnn/session_{}/model_{}.ckpt'.format(attempt, name))

            testing_batch = zip(range(0, len(teX), batch_size),
                                range(batch_size, len(teX) + 1, batch_size))

            total_accuracy = []
            for start, end in testing_batch:
                test_batch_accuracy = np.mean(np.argmax(teY[start:end], axis=1) ==
                                              sess.run(predict_op, feed_dict={X: teX[start:end],
                                                                              p_keep_conv: 1.0,
                                                                              p_keep_hidden: 1.0}))
                total_accuracy.append(test_batch_accuracy)

            test_accuracy_summary, m_accuracy = sess.run([merged, mean_accuracy],
                                                         feed_dict={batch_accuracies: total_accuracy})

            summary_writer.add_summary(test_accuracy_summary, i + 1)
            print("Epoch : {}, Test Acc: {}".format(i + 1, m_accuracy))
