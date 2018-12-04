import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage

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

    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],  # l1 shape=(?, 16, 16, 32)
                        strides=[1, 1, 1, 1], padding='VALID')


    l1 = tf.nn.dropout(l1, p_keep_conv)

    l3 = tf.reshape(l1, [-1, w_fc.get_shape().as_list()[0]])  # reshape to (?, 14x14x32)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w_fc))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx


def model5(X, p_keep_conv, p_keep_hidden):
    w = init_weights([3, 3, 3, 64])  # 3x3x3 conv, 32 outputs
    tf.summary.histogram("weights of first convolution layer 3x3x3x32", w)
    w_fc = init_weights([64 * 16 * 16, 625])  # FC 32 * 14 * 14 inputs, 625 outputs
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
    return l1a, pyx


batch_size = 128
    
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
Y = tf.placeholder("float", [None, 10], name='label')

# tf.summary.image('Input Image', tf.transpose(tf.reshape(X, shape=[batch_size, 3, 32, 32]), perm=[0, 2, 3, 1]))

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

        for epoch in range(15):
        
            # Initialize the training iterator to consume training data
            sess.run(train_init_op)    
            while True:
                # as long as the iterator has not hit the end, continue to consume training data
                try:
                    images, labels = sess.run(next_train_batch)
                    train_summary, _ = sess.run([merged, train_op], feed_dict={ X: images,
                                                                                Y: labels,
                                                                                p_keep_conv: 0.8, p_keep_hidden: 0.5})
                    break
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
                                                                p_keep_conv:1.0, 
                                                                p_keep_hidden:1.0}))
                    total_accuracy.append(test_batch_accuracy)
                    break
                except tf.errors.OutOfRangeError:
                    test_accuracy_summary, m_accuracy = sess.run([merged_testing_accuracy, mean_accuracy],
                                                feed_dict={batch_accuracies: total_accuracy})
                    summary_writer.add_summary(test_accuracy_summary, epoch)
                    # end of validation
                    break

            # Reinitialize test data to get top 9 patches from test data
            sess.run(test_init_op)
            images, labels = sess.run(next_test_batch)
            from heapq import heappush, heappushpop

            heap = []
            for image, label in zip(images, labels):
                fm = sess.run(l1a, feed_dict={X: [image]})
                if len(heap) < 9:
                    heappush(heap, (np.linalg.norm(fm[0]), image, fm[0]))
                else:
                    heappushpop(heap, (np.linalg.norm(fm[0]), image, fm[0]))

            top9 = sorted(heap, reverse=True)
            print("attempt", attempt)
            for distance, image, fm in top9:
                print(tf.shape(image))
                print(tf.shape(image[0]))
                print(tf.shape(image[0][0]))
                print("All", np.shape(np.array([image])))

                summary_op = tf.summary.image("model_projections",
                                              np.array([image]),
                                              max_outputs=1, family='normally')
                # Summary has to be evaluated (converted into a string) before adding to the writer
                summary_writer.add_summary(summary_op.eval(), 10)

                img = plt.imshow(toimage(image.reshape(3, 32, 32)), interpolation='nearest')
                plt.show()
                #summary_writer.add_summary(tf.summary.image('Image', image), image)
                #summary_writer.add_summary(tf.summary.image('Feature Map', fm), fm)

            summary_writer.flush()

            print("Epoch {}: testing accuracy: {}".format(epoch, test_accuracy_summary))
            
            if (epoch+1)%5 == 0:
                save_path = './logs/attempt_{}/model_checkpoint/model_{}_epoch{}.ckpt'.format(attempt, name, epoch+1)
                saver.save(sess, save_path)
                print("Saved model at {}".format(save_path))  
        # End of training
