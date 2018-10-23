import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt


# Function approximator model
def model(x, hidden_dim = 8):
    input_dim = 2 # we got x and y as our inputs
    output_dim = 1 # just one value as output
    stdev = 0.01
    with tf.variable_scope('FunctionApproximator'):
        w_h1 = tf.get_variable('w_h1', shape=[input_dim, hidden_dim], initializer=tf.random_normal_initializer(stddev=stdev))
        b_h1 = tf.get_variable('b_h1', shape=[hidden_dim], initializer=tf.constant_initializer(0.))

        z = tf.nn.tanh(tf.matmul(x, w_h1) + b_h1)

        w_o = tf.get_variable('w_o', shape=[hidden_dim, output_dim], initializer=tf.random_normal_initializer(stddev=stdev))

    return tf.matmul(z, w_o)


# Function we are approximating
def f(x, y):
    return np.cos(x + 6 * 0.35 * y) + 2 * 0.35 * x * y


def generate_data():
    mySet = []
    creating = True
    while creating:
        for x, y in zip(np.random.uniform(low=-1, high=1, size=(200,)),
                        np.random.uniform(low=-1, high=1, size=(200,))):
            if len(mySet) == 181:
                creating = False
                break
            dataInput = [x, y]
            mySet.append(dataInput)
    myList = list(mySet)
    labels = list([[f(x, y)] for x, y in myList])
    # return training inputs, training outputs, testing inputs, testing outputs
    return myList[:100], labels[:100], myList[100:], labels[100:]


# Generate our dataset
trX, trY, teX, teY = generate_data()

for size in [2, 8, 50]:
    tf.reset_default_graph()
    print("Training with {} number of hidden neurons".format(size))
    with tf.variable_scope('Graph') as scope:
        x = tf.placeholder("float", shape=[None, 2], name='inputs')
        y_true = tf.placeholder("float", shape=[None, 1], name='y_true')
        # output of our model
        y_pred = model(x, hidden_dim=size)
        with tf.variable_scope('Loss'):
            loss = tf.reduce_mean(tf.square(y_true - y_pred))
        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.02).minimize(loss)
        #train_op = tf.train.MomentumOptimizer(learning_rate=0.02, momentum=0.02).minimize(loss)
        #train_op = tf.train.RMSPropOptimizer(learning_rate=0.02).minimize(loss)
        predict_op = y_pred
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(10000):
            batch_size = 5
            for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trY) + 1, batch_size)):
                curr_loss, _ = sess.run([loss, train_op], feed_dict={x: trX[start:end], y_true: trY[start:end]})
                predicted = sess.run(predict_op, feed_dict={x: trX[start:end]})
            if sess.run(loss, feed_dict={x: trX, y_true: trY}) < 0.002:
                break
                    
            # print("Epoch {}: Loss: {}".format(i, curr_loss))

        print("{} Neurons results in a {} MSE, number of epochs: {}".format(size, sess.run(loss, feed_dict={x: trX,
                                                                                                            y_true: trY}), i))
        print("{} Neurons results in a {} MSE, number of epochs: {}".format(size, sess.run(loss, feed_dict={x: trX,
                                                                                                            y_true: trY}),
                                                                            i))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        u = np.linspace(-1, 1, 100)
        x1, y1 = np.meshgrid(u, u)
        #cs = ax.contour(x1, y1, f(x1, y1))
        #plt.clabel(cs, fontsize=10, colors=plt.cm.Reds(cs.norm(cs.levels)))
        #plt.colorbar(cs)
        #plt.show()

        com = np.vstack((x1.flatten(), y1.flatten())).T
        predicted = sess.run(predict_op, feed_dict={x: com})
        # print(test_loss)
        predicted = np.reshape(predicted, (-1, 100))

        actual = np.reshape(f(x1, y1), (100 * 100, -1))

        cs = ax.contour(x1, y1, predicted)
        plt.clabel(cs, fontsize=10, colors=plt.cm.Reds(cs.norm(cs.levels)))
        plt.colorbar(cs)

fig = plt.figure()
ax = fig.add_subplot(111)
cs = ax.contour(x1, y1, f(x1, y1))
plt.clabel(cs, fontsize=10, colors=plt.cm.Reds(cs.norm(cs.levels)))
plt.colorbar(cs)
plt.show()
        # plt.clabel(cs, fontsize=10, colors=plt.cm.Reds(cs.norm(cs.levels)))







