import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt


# Function approximator model
def model(x, hidden_dim=8):
    input_dim = 35
    output_dim = 31
    stdev = 0.1
    with tf.variable_scope('FunctionApproximator'):
        w_h1 = tf.get_variable('w_h1', shape=[input_dim, hidden_dim],
                               initializer=tf.random_normal_initializer(stddev=stdev))
        b_h1 = tf.get_variable('b_h1', shape=[hidden_dim], initializer=tf.random_normal_initializer(stddev=stdev))

        z = tf.nn.sigmoid(tf.matmul(x, w_h1) + b_h1)

        w_o = tf.get_variable('w_o', shape=[hidden_dim, output_dim],
                              initializer=tf.random_normal_initializer(stddev=stdev))

    return tf.matmul(z, w_o)


lines = open("./q2-patterns.txt").read().splitlines()
trX = {
    0: [],
    1: [],
    2: [],
    3: [],
}
trY = {
    0: np.zeros((31, 31)),
    1: np.zeros((31, 31)),
    2: np.zeros((31, 31)),
    3: np.zeros((31, 31)),
}

for noise in [0, 1, 2, 3]:
    for i in range(0, len(lines), 8):
        a = np.array(lines[i:i+7])
        a = list(''.join(a))
        for j in range(0, noise):
            index = random.randrange(0, len(a))
            a[index] = str(int(a[index], 2) ^ 1)

        trX[noise].append(np.array(a))
        trY[noise][int(i / 8)][int(i / 8)] = 1


# training on images with nosie of 0 and 3
# can do np.vstack((trX[0], trX[1], trX[2], trX[3])) to train on all noises
#trainingX = np.vstack((trX[0], trX[3]))
#trainingY = np.vstack((trY[0], trY[3]))
trainingX = trX[0]
trainingY = trY[0]
idx = np.random.permutation(len(trainingX))
batchSize = 1

data = []
labels = []

for j in range(0, len(idx), batchSize):
    data.append([])
    labels.append([])
    for k in range(0, batchSize):
        if j + k >= len(trainingX):
            break
        print(j + k)
        data[int(j/batchSize)].append(trainingX[idx[j + k]])
        labels[int(j/batchSize)].append(trainingY[idx[j + k]])
print(len(trainingX))
for size in [10, 15]:
    tf.reset_default_graph()
    print("Training with {} number of hidden neurons".format(size))
    with tf.variable_scope('Graph') as scope:
        x = tf.placeholder("float", shape=[None, 35], name='inputs')
        y_true = tf.placeholder("float", shape=[None, 31], name='y_true')
        # output of our model
        y_pred = model(x, hidden_dim=size)
        with tf.variable_scope('Loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))  # compute costs
        train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
        predict_op = tf.argmax(y_pred, 1)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            loss_list = []
            epochs = [i for i in range(500)]
            for i in range(500):
                average_loss = 0
                counter = 0
                for j in range(0, len(data)):
                    curr_loss, _ = sess.run([loss, train_op], feed_dict={x: data[j], y_true: labels[j]})
                    counter += 1
                    average_loss += curr_loss
                average_loss /= counter
                if i % 25 == 0:
                    print(i, curr_loss)
                loss_list.append(average_loss)
            
            plt.plot(epochs,loss_list)
            plt.ylabel('Training Error')
            plt.yscale('log')
            plt.show()
            for noise in [0, 1, 2, 3]:
                print(np.argmax(trY[noise], axis=1))
                print(sess.run(predict_op, feed_dict={x: trX[noise]}))

                print(i, "noise", noise, "Error", 1 - (np.mean(np.argmax(trY[noise], axis=1) ==
                        sess.run(predict_op, feed_dict={x: trX[noise]}))))









