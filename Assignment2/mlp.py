import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Todo, create a more flexible and reusable code later.
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.5))



def model(X, w_h1, w_o, bias):
    # Sigmoid is our activation function (making it non-linear)
    h1 = tf.nn.sigmoid(tf.matmul(X, w_h1) + bias) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h1, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

def f(x, y):
    return np.cos(x + 6 * 0.35 * y) + 2 * 0.35 * x * y


mySet = set()


creating = True
while creating:
    for x, y in zip(np.random.uniform(low=-1, high=1, size=(200,)),
                    np.random.uniform(low=-1, high=1, size=(200,))):
        if len(mySet) == 500:
            creating = False
            break
        mySet.add((x, y))

myList = list(mySet)
labels = list([[f(x, y)] for x, y in myList])


trX = myList[0:250]
trY = labels[0:250]

print("size training", len(trX))

teX = myList[250:500]
teY = labels[250:500]
print("size testing", len(teX))


fig = plt.figure()
ax = fig.add_subplot(111)
u = np.linspace(-1, 1, 5)
x, y = np.meshgrid(u, u)
print('x',x)

for size in [8]:
    print("Size", size)
    size_h1 = tf.constant(size, dtype=tf.int32)

    X = tf.placeholder("float", [None, 2])
    Y = tf.placeholder("float", [None, 1])

    w_h1 = init_weights([2, size_h1])  # create symbolic variables
    bias_h1 = tf.Variable(tf.random_normal([1], stddev=0.5))
    w_o = init_weights([size_h1, 1])  # once we remove w_h2, should fix the dimension for this layer

    py_x = model(X, w_h1, w_o, bias_h1)  # This returns us the outputs of our final layer in the model
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))  # compute costs
    # cost = tf.losses.mean_squared_error(labels=Y, predictions=py_x)
    cost = tf.reduce_mean(tf.square(py_x - Y))
    # cost = tf.reduce_mean(cost)

    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cost)  # construct an optimizer

    # here for 1b, we need to create 2 other optimizers, namely the Momentum and RMSProp Optimizers)
    predict_op = py_x

    saver = tf.train.Saver()

    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()


        u = np.linspace(-1, 1, 5)
        x, y = np.meshgrid(u, u)
        com = np.vstack((x.flatten(), y.flatten())).T

        predicted = sess.run(predict_op, feed_dict={X: com})
        predicted = np.reshape(predicted, (-1, 5))
        print(predicted)

        for i in range(1500):
            # This runs a single iteration (epoch)
            for start, end in zip(range(0, len(trX), 2), range(2, len(trX) + 1, 2)):
                sess.run(train_op, feed_dict={X: np.array(trX[start:end]), Y: trY[start:end]})


        predicted = sess.run(predict_op, feed_dict={X: teX})
        print("mse", np.mean((teY - predicted)**2))


        predicted = sess.run(predict_op, feed_dict={X: com})
        predicted = np.reshape(predicted, (-1, 5))
        print(predicted)
        print(com)

        cs = ax.contour(x, y, predicted)
        plt.clabel(cs, fontsize=10, colors=plt.cm.Reds(cs.norm(cs.levels)))
        plt.colorbar(cs)
        saver.save(sess, "mlp/session.ckpt")

#cs = ax.contour(x, y, f(x, y))
#plt.clabel(cs, fontsize=10, colors=plt.cm.Reds(cs.norm(cs.levels)))
#plt.colorbar(cs)
plt.show()