import tensorflow as tf
import numpy as np
import math


# Todo, create a more flexible and reusable code later.
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h1, w_o):
    # Sigmoid is our activation function (making it non-linear)
    h1 = tf.nn.sigmoid(tf.matmul(X, w_h1)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h1, w_o) # note that we dont take the softmax at the end because our cost fn does that for us


def f(x, y):
    return math.cos(x + 6*0.35 * y) + 2*0.35 * x * y

# PREPARATION FOR DATA:

# This example uses mnist, but we will generate data the function
# f(x,y)=cos(x+6*0.35y) + 2*0.35xy where x,y is element of [-1,1]
# The generated data should just be a list of tuples (x,y,label) or whatever,
# just need to make sure we have a way of identifying the input and its associated label.
# For now we can just pick an arbitrary number of data to generate, say 10,000 data.
# Randomize the data using np.shuffle or something
# and then split the data into 9:1 ratio for train:test sets (or whatever the answer is to the question we gotta ask).
# Now randomly take 10% (again not sure how much)
# of the data in the train set and put that into a validation set
# The idea is for each training iteration (epoch), we train on the train set,
# then validate the error on the validation set (Important, do NOT
# train on the validation set, e.g use the optimizer on it then back propagate it.
# Just test it like you would on a test set)


for size in [2, 8, 50]:
    size_h1 = tf.constant(size, dtype=tf.int32)

    X = tf.placeholder("float", [None, 2])
    Y = tf.placeholder("float", [None, 1])

    w_h1 = init_weights([2, size_h1])  # create symbolic variables
    w_o = init_weights([w_h1, 1])  # once we remove w_h2, should fix the dimension for this layer

    py_x = model(X, w_h1, w_o)  # This returns us the outputs of our final layer in the model

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))  # compute costs
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)  # construct an optimizer
    # here for 1b, we need to create 2 other optimizers, namely the Momentum and RMSProp Optimizers)
    predict_op = tf.argmax(py_x, 1)

    saver = tf.train.Saver()

    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()
        print(range(0, len(trX), 128))
        for i in range(3):
            # This runs a single iteration (epoch)
            for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
                # This feeds a single input into our neural network
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

            # After this for loop, we need to run a validation,
            # meaning we tell the session to run a predict_op on all the test set
            # calculating the MSE of our test set, at some point we will monitor this value
            # and stop training when it reaches a certain threshold
            # this is what is called early stopping
            # Need to create a table variable before to keep track of the MSE values for the 3 different network sizes
            print(i, np.mean(np.argmax(teY, axis=1) ==
                             sess.run(predict_op, feed_dict={X: teX})))
        saver.save(sess, "mlp/session.ckpt")
