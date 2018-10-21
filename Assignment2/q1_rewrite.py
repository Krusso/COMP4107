import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

#Function approximator model
def model(x, hidden_dim = 8):
    input_dim = 2 #we got x and y as our inputs
    output_dim = 1 #just one value as output
    
    with tf.variable_scope('FunctionApproximator'):
        w_h1 = tf.get_variable('w_h1', shape=[input_dim, hidden_dim], initializer=tf.random_normal_initializer(stddev=1))
        b_h1 = tf.get_variable('b_h1', shape=[hidden_dim], initializer=tf.constant_initializer(0.))

        z = tf.nn.sigmoid(tf.matmul(x, w_h1) + b_h1)

        w_o = tf.get_variable('w_o', shape=[hidden_dim, output_dim], initializer=tf.random_normal_initializer(stddev=1))

    return tf.matmul(z, w_o)

#Function we are approximating
# def f(input):
#     print(input[0], input[1])
#     return np.cos(input[0] + 6 * 0.35 * input[1]) + 2 * 0.35 * input[0] * input[1]
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
    #return training inputs, training outputs, testing inputs, testing outputs
    return myList[:100], labels[:100], myList[100:], labels[100:]

trX, trY, teX, teY = generate_data()
print(trX)
with tf.variable_scope('Graph') as scope:
    x = tf.placeholder("float", shape=[None, 2], name='inputs')
    y_true = tf.placeholder("float", shape=[None, 1], name='y_true')
    
    #output of our model
    y_pred = model(x, hidden_dim = 8)
    
    with tf.variable_scope('Loss'):
        loss = tf.reduce_mean(tf.square(y_true - y_pred))

    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
    predict_op = y_pred
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #training
    print(len(trX))
    for i in range(100):
        for start, end in zip(range(0, len(trX), 10), range(10, len(trY) + 1, 10)):
            curr_loss, _ = sess.run([loss, train_op], feed_dict={x:trX[start:end], y_true: trY[start:end]})

        print("Epoch {}: Loss: {}".format(i, curr_loss))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    u = np.linspace(-1, 1, 5)
    x, y = np.meshgrid(u, u)

    cs = ax.contour(x, y, f(x, y))
    plt.clabel(cs, fontsize=10, colors=plt.cm.Reds(cs.norm(cs.levels)))
    plt.colorbar(cs)
    plt.show()


    # print(x, y)
    com = np.vstack((x.flatten(), y.flatten())).T
    teX = []
    for i in range(len(com)):
        print(com[i])
        teX.append(com[i])
    print('teX',teX)
    print('com',com)
    predicted = sess.run(predict_op, feed_dict={x: teX})
    # predicted = np.reshape(predicted, (-1, 5))
    # print(predicted)
    # print(x)
    # for i in com:
    #     print(f(i[0],i[1]))
    cs = ax.contour(x, y, predicted)
    # plt.clabel(cs, fontsize=10, colors=plt.cm.Reds(cs.norm(cs.levels)))







