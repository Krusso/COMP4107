import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage
import random

def parse_example(example):
    features = {'image_raw': tf.FixedLenFeature((), tf.string, default_value=""),
                'label': tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(example, features)
    
    image = tf.decode_raw(parsed_features['image_raw'], tf.float32)
    image = tf.reshape(image, [32, 32, 3])
    
    label = tf.cast(parsed_features['label'], tf.int64)
    return image, tf.one_hot(label, 10)


def input_fn(filenames, shuffle_buff=1500, batch_size=128):
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

    l1 = tf.nn.dropout(l1, p_keep_conv)

    l3 = tf.reshape(l1, [-1, w_fc.get_shape().as_list()[0]])  # reshape to (?, 14x14x32)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w_fc))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return l1a, pyx, 32


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
    return l1a, pyx, 32


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
    return l1a, pyx, 32


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
    return l1a, pyx, 32


def model5(X, p_keep_conv, p_keep_hidden):
    w_1 = init_weights([3, 3, 3, 32])  # 3x3x3 conv, 32 outputs
    tf.summary.histogram("weights of first convolution layer 3x3x3x32", w_1)

    w_2 = init_weights([3, 3, 32, 32])  # 3x3x3 conv, 32 outputs
    tf.summary.histogram("weights of second convolution layer 3x3x32x32", w_2)

    w_3 = init_weights([3, 3, 32, 32])  # 3x3x3 conv, 32 outputs
    tf.summary.histogram("weights of third convolution layer 3x3x32x32", w_3)

    w_fc = init_weights([32 * 8 * 8, 625])  # FC 32 * 14 * 14 inputs, 625 outputs
    tf.summary.histogram("weights of fully connected 625 neuron first layer", w_fc)
    w_o = init_weights([625, 10])  # FC 625 inputs, 10 outputs (labels)
    tf.summary.histogram("weights of 10 neuron output layer", w_o)

    l1a = tf.nn.relu(tf.nn.conv2d(X, w_1,  # X=(128,32,32,3) l1a shape=(?, 32, 32, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    print('l1a shape,',l1a.shape)
    l2a = tf.nn.relu(tf.nn.conv2d(l1a, w_2,  #l1a shape=(?, 32, 32, 32) l2a shape=(?, 32, 32, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    print('l2a shape', l2a.shape)
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],  # l1 shape=(?, 8, 8, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)
    print('l2.shape',l2.shape)
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w_3, 
                                 strides=[1,1,1,1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1,2,2,1],
                            strides=[1,2,2,1], padding='SAME')

    l3 = tf.reshape(l3, [-1, w_fc.get_shape().as_list()[0]])  # reshape to (?, 14x14x32)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w_fc))
    l4 = tf.nn.dropout(l4, p_keep_hidden)
    print("p4 shape", l4.shape)
    pyx = tf.matmul(l4, w_o)
    print(pyx.shape)
    return l1a, pyx, 32


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

feature_map_image = tf.placeholder("float", [None, 32, 32, 1])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

attempt = 0

while True:
    if os.path.isdir('./logs/attempt_{}'.format(attempt)):
        attempt += 1
    else:
        break


for name, model in list([("model 5", model1(X, p_keep_conv, p_keep_hidden))]):
    l1a, py_x, features = model

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
                    # print(images)
                    train_summary, _ = sess.run([merged, train_op], feed_dict={ X: images,
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
                                                                p_keep_conv:1.0, 
                                                                p_keep_hidden:1.0}))
                    total_accuracy.append(test_batch_accuracy)
                    
                except tf.errors.OutOfRangeError:
                    test_accuracy_summary, m_accuracy = sess.run([merged_testing_accuracy, mean_accuracy],
                                                feed_dict={batch_accuracies: total_accuracy})

                    summary_writer.add_summary(test_accuracy_summary, epoch)
                    # end of validation
                    print("Epoch {}: testing accuracy: {}".format(epoch, test_accuracy_summary))
            
                    if (epoch+1)%5 == 0:
                        save_path = './logs/attempt_{}/model_checkpoint/model_{}_epoch{}.ckpt'.format(attempt, name, epoch+1)
                        saver.save(sess, save_path)
                        print("Saved model at {}".format(save_path))  
                    break

            # Reinitialize test data to get top 9 patches from test data
            sess.run(test_init_op)
            images, labels = sess.run(next_test_batch)
            from heapq import heappush, heappushpop

            heap = []
            indices = [i for i in range(features)]
            indices = random.sample(indices, min(9, features))

            for image, label in zip(images, labels):
                fm = sess.run(l1a, feed_dict={X: [image]})
                fm = fm.transpose(3, 1, 2, 0)

                for index in indices:
                    if len(heap) < 9:
                        heappush(heap, (np.linalg.norm(fm[index]), image, fm[index], index))
                    else:
                        heappushpop(heap, (np.linalg.norm(fm[index]), image, fm[index], index))


            
            top9 = sorted(heap, reverse=True)

            fmc = 0

            for distance, image, fm, index in top9:
                fmc += 1
                summary_op = tf.summary.image("image causing #{} highest activation for feature map #{}".
                                              format(fmc, index),
                                              np.array([image.reshape(3, 32, 32).transpose(1, 2, 0)]),
                                              max_outputs=1, family="image")

                summary_writer.add_summary(summary_op.eval(), epoch)

                summary_op = tf.summary.image("#{} highest activation for feature map #{}"
                                              .format(fmc, index),
                                              np.array([fm]),
                                              max_outputs=1, family='feature_map')

                summary_writer.add_summary(summary_op.eval(), epoch)

                cmap = plt.get_cmap('jet')
                rgba_img = cmap(fm.reshape(32, 32))

                rgb = np.delete(rgba_img, 3, 2)

                summary_op = tf.summary.image("#{} highest activation for feature heatmap #{}"
                                              .format(fmc, index),
                                              np.array([rgb]),
                                              max_outputs=1, family='feature_map')

                summary_writer.add_summary(summary_op.eval(), epoch)



                # img = plt.imshow(toimage(image.reshape(3, 32, 32)), interpolation='nearest')
                # plt.show()
                # #summary_writer.add_summary(tf.summary.image('Image', image), image)
                # #summary_writer.add_summary(tf.summary.image('Feature Map', fm), fm)

            summary_writer.flush()


        # # End of training
