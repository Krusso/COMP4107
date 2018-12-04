import tensorflow as tf
import numpy as np
import os
import tarfile
from urllib.request import urlretrieve
from scipy.misc import toimage
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from math import sqrt

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
    train_images, test_images = images[:1000], images[1000:2000]
    train_labels, test_labels = labels[:1000], labels[1000:2000]

    def _onehot(integer_labels):
        """Return matrix whose rows are onehot encodings of integers."""
        n_rows = len(integer_labels)
        n_cols = integer_labels.max() + 1
        onehot = np.zeros((n_rows, n_cols), dtype='uint8')
        onehot[np.arange(n_rows), integer_labels] = 1
        return onehot

    return train_images, _onehot(train_labels), \
        test_images, _onehot(test_labels)

def parse_example(example):
    features = {'image_raw': tf.FixedLenFeature((), tf.string, default_value=""),
                'label': tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(example, features)
    
    image = tf.decode_raw(parsed_features['image_raw'], tf.float32)
    image = tf.reshape(image, [32, 32, 3])
    
    label = tf.cast(parsed_features['label'], tf.int64)
    return image, tf.one_hot(label, 10)

def input_fn(filenames, shuffle_buff=15000, batch_size=100):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.shuffle(shuffle_buff)
    dataset = dataset.map(lambda example: parse_example(example))
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    return dataset

def put_kernels_on_grid (kernel, pad = 1):

    '''Visualize conv. filters as an image (mostly for the 1st layer).
    Arranges filters into a grid, with some paddings between adjacent filters.
    Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    pad:               number of black pixels around each filter (between them)
    Return:
    Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
    '''
    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of filters')
                return (i, int(n / i))
    (grid_Y, grid_X) = factorization (kernel.shape[3])
    print ('grid: %d = (%d, %d)' % (kernel.shape[3], grid_Y, grid_X))

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    kernel = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + 2 * pad
    X = kernel.get_shape()[1] + 2 * pad

    channels = kernel.get_shape()[2]

    # put NumKernels to the 1st dimension
    x = tf.transpose(x, (3, 0, 1, 2))
    # organize grid on Y axis
    x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x = tf.transpose(x, (0, 2, 1, 3))
    # organize grid on X axis
    x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x = tf.transpose(x, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x = tf.transpose(x, (3, 0, 1, 2))

    # scaling to [0, 255] is not necessary for tensorboard
    return x


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

    # tf.summary.image('Layer 1 Max Pool', l1)

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

    #tf.summary.image('layer 1', l1a)

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
    w = init_weights([3, 3, 3, 32])  # 3x3x3 conv, 32 outputs
    tf.summary.histogram("weights of first convolution layer 3x3x3x32", w)
    w_fc = init_weights([32 * 31 * 31, 625])  # FC 32 * 14 * 14 inputs, 625 outputs
    tf.summary.histogram("weights of fully connected 625 neuron first layer", w_fc)
    w_o = init_weights([625, 10])  # FC 625 inputs, 10 outputs (labels)
    tf.summary.histogram("weights of 10 neuron output layer", w_o)

    l1a = tf.nn.relu(tf.nn.conv2d(X, w,  # l1a shape=(?, 32, 32, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    
# # Visualize the activation patch
    V = tf.slice(l1a, (0, 0, 0, 0), (1, -1, -1, -1), name='slice_first_input')
#     V = tf.reshape(V, (32, 32, 32))

#     # Reorder so the channels are in the first dimension, x and y follow.
#     V = tf.transpose(V, (2, 0, 1))
#     # Bring into shape expected by image_summary
#     V = tf.reshape(V, (-1, 32, 32, 1))

    V = put_kernels_on_grid(V)
    tf.summary.image("Layer 1 Activation", V)

    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],  # l1 shape=(?, 16, 16, 32)
                        strides=[1, 1, 1, 1], padding='VALID')


    l1 = tf.nn.dropout(l1, p_keep_conv)

    l3 = tf.reshape(l1, [-1, w_fc.get_shape().as_list()[0]])  # reshape to (?, 14x14x32)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w_fc))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx
tf.extract_image_patches

def model5(X, p_keep_conv, p_keep_hidden):
    w = init_weights([3, 3, 3, 64])  # 3x3x3 conv, 32 outputs
    tf.summary.histogram("weights of first convolution layer 3x3x3x32", w)
    w_fc = init_weights([64 * 16 * 16, 625])  # FC 32 * 14 * 14 inputs, 625 outputs
    tf.summary.histogram("weights of fully connected 625 neuron first layer", w_fc)
    w_o = init_weights([625, 10])  # FC 625 inputs, 10 outputs (labels)
    tf.summary.histogram("weights of 10 neuron output layer", w_o)

    l1a = tf.nn.relu(tf.nn.conv2d(X, w,  # l1a shape=(?, 32, 32, 32)
                                strides=[1, 1, 1, 1], padding='SAME'))

    # V = tf.slice(l1a, (0, 0, 0, 0), (1, -1, -1, -1), name='slice_first_input')
    # print('SHAPE of V: ', V.shape)
    # V = put_kernels_on_grid(V)
    # tf.summary.image("Layer 1 Activation", V)
    # activation_summary = tf.contrib.layers.summarize_activation(l1a)
    deconvl1 = tf.nn.conv2d_transpose(l1a, w, output_shape=[64,32,32,3], strides=[1,1,1,1], name='Deconv1')
    
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],  # l1 shape=(?, 16, 16, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l3 = tf.reshape(l1, [-1, w_fc.get_shape().as_list()[0]])  # reshape to (?, 14x14x32)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w_fc))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return deconvl1, pyx



def visualize_activation_layer(summary_name, layer, l, w, channels):
    V = tf.slice(layer, (0, 0, 0, 0), (1, -1, -1, -1), name='slice_first_input') #todo: need to get top 9 patches some how
    V = tf.reshape(V, (32, 32, 32))

    # Reorder so the channels are in the first dimension, x and y follow.
    V = tf.transpose(V, (2, 0, 1))
    # Bring into shape expected by image_summary
    V = tf.reshape(V, (-1, 32, 32, 1))
    tf.summary.image(name, V)

batch_size = 128
    
train_set = input_fn(filenames=["distorted_images/train_set.tfrecords"],
                                batch_size=batch_size)

test_set = input_fn(filenames=["distorted_images/test_set.tfrecords"],
                                batch_size=batch_size)   
#Training iterators                            
train_iterator = train_set.make_initializable_iterator()
test_iterator = test_set.make_initializable_iterator()

#Pipeline for feeding into the net, gets the next batch in dataset
next_train_batch = train_iterator.get_next() 
next_test_batch = test_iterator.get_next()

#Used to reset the iterator
train_init_op = train_iterator.initializer
test_init_op = test_iterator.initializer


# trX, trY, teX, teY = cifar10(path='./tmp')
# for i in range(4):
#     img = plt.imshow(toimage(trX[i].reshape(3, 32, 32)), interpolation='nearest')
#     plt.show()
#     img = plt.imshow(toimage(trX[i].reshape(3, 32, 32)), interpolation='bicubic')
#     print(trX[i])
#     plt.show()
#     print(trY[i])
#     print(trX[i].shape)
    

X = tf.placeholder("float", [None, 32, 32, 3], name='image')
Y = tf.placeholder("float", [None, 10], name='label')

tf.summary.image('Input Image', tf.transpose(tf.reshape(X, shape=[batch_size, 3, 32, 32]), perm=[0, 2, 3, 1]))
# tf.summary.image('Input Image Other', toimage(tf.reshape(X, [batch_size, 3, 32, 32])))

feature_map_image = tf.placeholder("float", [None, 32, 32, 1])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

attempt = 0

while True:
    if os.path.isdir('./logs/attempt_{}'.format(attempt)):
        attempt += 1
    else:
        break


for name, model in list([("model 5", model5(X, p_keep_conv, p_keep_hidden))]):
    l1a, py_x = model

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(py_x, 1)

    batch_accuracies = tf.placeholder("float", [None])
    mean_accuracy = tf.reduce_mean(tf.cast(batch_accuracies, tf.float32))

    # Launch the graph in a session
    with tf.Session() as sess:
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        
        testing_accuracy = tf.summary.scalar('accuracy', mean_accuracy)
        merged_testing_accuracy = tf.summary.merge([testing_accuracy])
        image_summary = tf.summary.image("image", X, max_outputs = 9)
        merged_image = tf.summary.merge([image_summary])

        result_dir = './logs/attempt_{}/{}'.format(attempt, name)
        summary_writer = tf.summary.FileWriter(result_dir, graph=sess.graph)
        
        feature_map_summary = tf.summary.image("layer 1 activation map", feature_map_image, max_outputs=9)
        feature_map_merged = tf.summary.merge([feature_map_summary])

        # you need to initialize all variables
        tf.global_variables_initializer().run()
#############
        for epoch in range(15):
        
        #Initialize the training iterator to consume training data
            sess.run(train_init_op)    
            while True:
                #as long as the iterator has not hit the end, continue to consume training data
                try:
                    images, labels = sess.run(next_train_batch)
                    train_summary, _ = sess.run([merged, train_op], feed_dict={ X: images,
                                                                                Y: labels,
                                                                                p_keep_conv: 0.8, p_keep_hidden: 0.5})
                except tf.errors.OutOfRangeError:
                    #end of training epoch
                    summary_writer.add_summary(train_summary, epoch)
                    break
            #Initialize the validation iterator to consume validation data
            sess.run(test_init_op)
            total_accuracy = []
            while True:
                try:
                    images, labels = sess.run(next_test_batch)
                    test_batch_accuracy = np.mean(np.argmax(labels, axis=1) == 
                                                    sess.run(predict_op, feed_dict={X: images,
                                                                Y: labels, 
                                                                p_keep_conv:1.0, 
                                                                p_keep_hidden:1.0}))
                    total_accuracy.append(test_batch_accuracy)
                                                        
                except tf.errors.OutOfRangeError:
                    test_accuracy_summary, m_accuracy = sess.run([merged_testing_accuracy, mean_accuracy],
                                                feed_dict={batch_accuracies: total_accuracy})
                    summary_writer.add_summary(test_accuracy_summary, epoch)
                    #end of validation
                    break

            #                     total_accuracy = []
            # for start, end in testing_batch:
            #     test_batch_accuracy = np.mean(np.argmax(teY[start:end], axis=1) ==
            #                                   sess.run(predict_op, feed_dict={X: teX[start:end],
            #                                                                   p_keep_conv: 1.0,
            #                                                                   p_keep_hidden: 1.0}))
            #     total_accuracy.append(test_batch_accuracy)

            # test_accuracy_summary, m_accuracy = sess.run([merged_testing_accuracy, mean_accuracy],
            #                                              feed_dict={batch_accuracies: total_accuracy})
            print("Epoch {}: testing accuracy: {}".format(epoch, test_accuracy_summary))
            
            if (epoch+1)%5 == 0:
                save_path = './logs/attempt_{}/model_checkpoint/model_{}_epoch{}.ckpt'.format(attempt, name, epoch+1)
                saver.save(sess, save_path)
                print("Saved model at {}".format(save_path))  
        #End of training  


###############
        # for i in range(15):
        #     training_batch = zip(range(0, len(trX), batch_size),
        #                          range(batch_size, len(trX) + 1, batch_size))

        #     count = 1
        #     for start, end in training_batch:
        #         print('batch number {}'.format(count))
        #         count += 1
        #         # image_batch = trX[start:end]
        #         # distorted_images = sess.run(tf.image.random_flip_left_right(image_batch))
        #         # distorted_images = sess.run(tf.image.random_flip_up_down(distorted_images))
        #         # distort_image = tf.image.random_flip_up_down(image_batch)
        #         # distort_image = tf.image.random_flip_left_right(distort_image)
        #         # print(len(distorted_images))
        #         # print(distorted_images.shape)

        #         summary, _ = sess.run([merged, train_op], feed_dict={X: trX[start:end], Y: trY[start:end],
        #                                       p_keep_conv: 0.8, p_keep_hidden: 0.5})

        #         #visualize feature maps for one image after a training epoch
        #         filteredImage = sess.run(l1a, feed_dict={X: trX[0].reshape(1,32,32,3), p_keep_conv: 1.0, p_keep_hidden: 1.0}) #(1,32,32,64)
                
        #         filteredImage = sess.run(tf.transpose(filteredImage, (1,2,0,3))) #(64,32,32,1)
        #         print(filteredImage.shape)
        #         a_maps = put_kernels_on_grid(filteredImage)
        #         print("shape of a map,", a_maps.shape)
        #         print("filtered image shape:", filteredImage.shape[3])
        #         print(tf.size(filteredImage))
        #         filteredImage = sess.run(tf.transpose(filteredImage, (3,2,1,0))) #(64,32,32,1)
        #         print(filteredImage.shape)
        #         # tf.summary.image('Input Image', tf.transpose(tf.reshape(X, shape=[batch_size, 3, 32, 32]), perm=[0, 2, 3, 1]))
        #         filteredImage = sess.run(tf.transpose(tf.reshape(filteredImage, shape=[64, 1, 32, 32]), perm=[0, 2, 3, 1]))
        #         # filteredImage = sess.run(tf.transpose(filteredImage, (0,3,1,2)))
        #         # filteredImage = tf.reshape(filteredImage, (64,32,32,1))
        #         print(filteredImage.shape)
        #         fm_summary = sess.run(feature_map_summary, feed_dict={feature_map_image: filteredImage})
        #         summary_writer.add_summary(fm_summary)
        #         # for i in range(filteredImage.shape[3]):
        #         #     img1 = filteredImage[:,:,:,i]
        #         #     print(img1.shape)
        #         #     plt.imshow(filteredImage[:,:,:,i].reshape(32,32))
        #         #     plt.show()
                
        #     summary_writer.add_summary(summary)#just add the last batch of images



            # saver.save(sess, './logs/attempt_{}/model_checkpoint/model_{}.ckpt'.format(attempt, name))

            # testing_batch = zip(range(0, len(teX), batch_size),
            #                     range(batch_size, len(teX) + 1, batch_size))

            # total_accuracy = []
            # for start, end in testing_batch:
            #     test_batch_accuracy = np.mean(np.argmax(teY[start:end], axis=1) ==
            #                                   sess.run(predict_op, feed_dict={X: teX[start:end],
            #                                                                   p_keep_conv: 1.0,
            #                                                                   p_keep_hidden: 1.0}))
            #     total_accuracy.append(test_batch_accuracy)

            # test_accuracy_summary, m_accuracy = sess.run([merged_testing_accuracy, mean_accuracy],
            #                                              feed_dict={batch_accuracies: total_accuracy})

            # summary_writer.add_summary(test_accuracy_summary, i + 1)
            # print("Epoch : {}, Test Acc: {}".format(i + 1, m_accuracy))
