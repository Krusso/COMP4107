import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

question = "c"

if len(sys.argv) == 2:
    if sys.argv[1] == "a":
        question = "a"
    elif sys.argv[1] == "b":
        question = "b"
    elif sys.argv[1] == "c":
        question = "c"
    else:
        question = "a"


# Function approximator model
def model(x, hidden_dim=8):
    input_dim = 2  # we got x and y as our inputs
    output_dim = 1  # just one value as output
    if question == "a":
        stdev = 1
    elif question == "b":
        stdev = 0.01
    else:
        stdev = 0.8

    with tf.variable_scope('FunctionApproximator'):
        w_h1 = tf.get_variable('w_h1', shape=[input_dim, hidden_dim],
                               initializer=tf.random_normal_initializer(stddev=stdev))
        b_h1 = tf.get_variable('b_h1', shape=[hidden_dim], initializer=tf.constant_initializer(0.))

        z = tf.nn.tanh(tf.matmul(x, w_h1) + b_h1)

        w_o = tf.get_variable('w_o', shape=[hidden_dim, output_dim],
                              initializer=tf.random_normal_initializer(stddev=stdev))

    return tf.matmul(z, w_o)


# Function we are approximating
def f(x, y):
    return np.cos(x + 6 * 0.35 * y) + 2 * 0.35 * x * y


def farray(x):
    return np.cos(x[0] + 6 * 0.35 * x[1]) + 2 * 0.35 * x[0] * x[1]


def generate_data():
    u = np.linspace(-1, 1, 10)
    x1, y1 = np.meshgrid(u, u)

    trX = np.vstack((x1.flatten(), y1.flatten())).T
    np.random.shuffle(trX)
    trY = [[f(x[0], x[1])] for x in trX]

    u = np.linspace(-1, 1, 9)
    x1, y1 = np.meshgrid(u, u)

    teX = np.vstack((x1.flatten(), y1.flatten())).T
    np.random.shuffle(teX)
    teY = [[f(x[0], x[1])] for x in teX]

    mySet = []
    creating = True
    while creating:
        for x, y in zip(np.random.uniform(low=-1, high=1, size=(200,)),
                        np.random.uniform(low=-1, high=1, size=(200,))):
            if len(mySet) == 200:
                creating = False
                break
            dataInput = [x, y]
            mySet.append(dataInput)
    myList = list(mySet)
    labels = list([[f(x, y)] for x, y in myList])

    return trX, trY, teX, teY, myList, labels


trX, trY, teX, teY, vX, vY = generate_data()

if question == "a":
    u = np.linspace(-1, 1, 9)
    x1, y1 = np.meshgrid(u, u)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs1 = ax.contour(x1, y1, f(x1, y1))
    plt.clabel(cs1, fontsize=10, colors=plt.cm.Reds(cs1.norm(cs1.levels)))
    plt.colorbar(cs1)

    table = [np.zeros(3), np.zeros(3), np.zeros(3)]
    values = [np.zeros((9, 9)), np.zeros((9, 9)), np.zeros((9, 9))]
    index = -1
    repeatSize = 10
    for size in [2, 8, 50]:
        index += 1
        for repeat in range(repeatSize):
            while True:
                tf.reset_default_graph()
                print("Training with {} number of hidden neurons".format(size))
                with tf.variable_scope('Graph') as scope:
                    x = tf.placeholder("float", shape=[None, 2], name='inputs')
                    y_true = tf.placeholder("float", shape=[None, 1], name='y_true')
                    # output of our model
                    y_pred = model(x, hidden_dim=size)
                    with tf.variable_scope('Loss'):
                        loss = tf.reduce_mean(tf.square(y_true - y_pred))
                    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)
                    predict_op = y_pred

                saver = tf.train.Saver()
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())

                    converged = False
                    epoch = "Didnt converge"
                    for i in range(2000):
                        batch_size = 30
                        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trY) + 1, batch_size)):
                            sess.run([loss, train_op], feed_dict={x: trX[start:end], y_true: trY[start:end]})
                        if i % 500 == 0:
                            print(sess.run(loss, feed_dict={x: trX, y_true: trY}))
                        if sess.run(loss, feed_dict={x: trX, y_true: trY}) < 0.02 and not converged:
                            print("Converged at epoch", i, "with MSE of", sess.run(loss, feed_dict={x: trX, y_true: trY}))
                            converged = True
                            epoch = i + 1

                    print("Epoch", epoch == "Didnt converge")
                    if epoch == "Didnt converge":
                        continue

                    print("{} Neurons results in a {} MSE, after {} epochs".format(
                        size,
                        sess.run(loss, feed_dict={x: teX, y_true: teY}), i))

                    print("Before", table[index])
                    table[index] = table[index] + np.array([size, sess.run(loss, feed_dict={x: teX, y_true: teY}), epoch])
                    print("After", table[index])

                    com = np.vstack((x1.flatten(), y1.flatten())).T
                    predicted = sess.run(predict_op, feed_dict={x: com})
                    predicted = np.reshape(predicted, (-1, 9))

                    values[index] = values[index] + predicted
                    break

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cs = ax.contour(x1, y1, values[index] / repeatSize, levels=cs1.levels)
        plt.clabel(cs, fontsize=10, colors=plt.cm.Reds(cs1.norm(cs1.levels)))
        plt.colorbar(cs)

    values.append(f(x1, y1) * repeatSize)

    print("%s \t %s" % ("Size", "average epochs to convergence"))
    for i in table:
        print("%s \t %s" % (i[0] / repeatSize, i[2] / repeatSize))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    j = 0
    c = ['g', 'r', 'b', 'k']
    legend = []
    for i in values:
        a = ax.contour(x1, y1, i / repeatSize, levels=cs1.levels, colors=c[j])
        h, _ = a.legend_elements()
        legend.append(h)
        j = j + 1
    ax.legend([legend[0][0], legend[1][0], legend[2][0], legend[3][0]], ['2', '8', '50', 'target'])
    plt.show()


elif question == "b":
    table = [np.zeros(2), np.zeros(2), np.zeros(2)]
    mses = [np.zeros((100, 2)), np.zeros((100, 2)), np.zeros((100, 2))]
    cpus = [np.zeros((100, 2)), np.zeros((100, 2)), np.zeros((100, 2))]
    ac = [np.zeros(3), np.zeros(3), np.zeros(3)]
    acC = [np.zeros(3), np.zeros(3), np.zeros(3)]
    index = -1
    repeatSize = 10
    traingingStyles = ["traingd", "traingdm", "traingrms"]
    for train in [(0, tf.train.GradientDescentOptimizer(learning_rate=0.02)),
                  (1, tf.train.MomentumOptimizer(learning_rate=0.02, momentum=0.02)),
                  (2, tf.train.RMSPropOptimizer(learning_rate=0.02))]:
        index += 1
        for repeat in range(repeatSize):
            while True:
                tf.reset_default_graph()
                mse = []
                cpu = []
                print("Training with {} number of hidden neurons".format(8))
                with tf.variable_scope('Graph') as scope:
                    x = tf.placeholder("float", shape=[None, 2], name='inputs')
                y_true = tf.placeholder("float", shape=[None, 1], name='y_true')
                # output of our model
                y_pred = model(x, hidden_dim=8)
                with tf.variable_scope('Loss'):
                    loss = tf.reduce_mean(tf.square(y_true - y_pred))
                train_op = train[1].minimize(loss)
                predict_op = y_pred

                saver = tf.train.Saver()
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())

                    converged = False
                    epoch = "Didnt converge"
                    for i in range(100):
                        batch_size = 1
                        start_time = time.process_time()
                        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trY) + 1, batch_size)):
                            sess.run(train_op, feed_dict={x: trX[start:end], y_true: trY[start:end]})

                        end = time.process_time()
                        error = sess.run(loss, feed_dict={x: trX, y_true: trY})
                        mse.append([train[0], error])
                        cpu.append([train[0], end - start_time])

                        if error < 0.02 and not converged:
                            print("Converged at epoch", i)
                            converged = True
                            epoch = i + 1
                            acC[index] += np.array([train[0], error, sess.run(loss, feed_dict={x: teX, y_true: teY})])

                    print("Epoch", epoch == "Didnt converge")
                    if epoch == "Didnt converge":
                        continue

                    cpus[index] += cpu
                    mses[index] += mse
                    table[index] = table[index] + np.array([train[0], epoch])
                    print("Training method {} resulted in a {} MSE, number of epochs: {}".format(
                        train_op,
                        sess.run(loss, feed_dict={x: trX, y_true: trY}), i))
                    ac[index] += np.array([train[0], sess.run(loss, feed_dict={x: trX, y_true: trY}),
                               sess.run(loss, feed_dict={x: teX, y_true: teY})])
                    break

    print("%s \t %s" % ("Training Style", "epochs to convergence"))
    print("%s \t %s" % ("traingd", table[0][1] / repeatSize))
    print("%s \t %s" % ("traingdm", table[1][1] / repeatSize))
    print("%s \t %s" % ("traingrms", table[2][1] / repeatSize))


    def show_graph(results, xaxis, yaxis):
        for result in results:
            y = [i[1] / repeatSize for i in result]
            x = [j + 1 for j in range(len(result))]
            plt.plot(x, y, label=traingingStyles[(result[0][0] / repeatSize).astype(int)], linestyle='--', marker='o')
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)
        plt.legend()
        plt.show()


    def show_graph_bar(results, xaxis, yaxis):
        x = np.array([j + 1 for j in range(len(results[0]))])
        y = [i[1] / repeatSize for i in results[0]]
        y1 = [i[1] / repeatSize for i in results[1]]
        y2 = [i[1] / repeatSize for i in results[2]]
        ax = plt.subplot(111)
        ax.bar(x - 0.3, y, width=0.3, color='b', align='center', label="traingd")
        ax.bar(x, y1, width=0.3, color='r', align='center', label="traingdm")
        ax.bar(x + 0.3, y2, width=0.3, color='g', align='center', label="traingrms")
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)
        plt.legend()
        plt.show()

    show_graph(mses, "Epoch", "MSE")
    show_graph_bar(cpus, "Epoch", "CPU Time per epoch (s)")

    print("The training method with the best accuracy wrt to testing data at the end of the 100 epochs is",
          traingingStyles[(sorted(ac, key=lambda q: q[2])[0][0] / repeatSize).astype(int)], "with an mse of", sorted(ac, key=lambda q: q[2])[0][2] / repeatSize)

    print("The training method with the best accuracy wrt to testing data when training error is reached is",
          traingingStyles[(sorted(acC, key=lambda q: q[2])[0][0] / repeatSize).astype(int)], "with an mse of", sorted(acC, key=lambda q: q[2])[0][2] / repeatSize)

if question == "c":
    u = np.linspace(-1, 1, 9)
    x1, y1 = np.meshgrid(u, u)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    values = [[], []]

    trainings = [[], [], []]

    layerSize = [7, 8, 10, 15, 25, 50]
    mses = [0 for i in range(len(layerSize))]
    idx = 0
    numExperiments = 10
    for size in layerSize:
        # take the average RMSE over 5 experiments for each hidden layer size
        
        for i in range(numExperiments):
            tf.reset_default_graph()
            with tf.variable_scope('Graph') as scope:
                x = tf.placeholder("float", shape=[None, 2], name='inputs')
                y_true = tf.placeholder("float", shape=[None, 1], name='y_true')
                # output of our model
                y_pred = model(x, hidden_dim=size)
                with tf.variable_scope('Loss'):
                    loss = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
                train_op = tf.train.RMSPropOptimizer(learning_rate=0.005, centered=True, momentum=0.1).minimize(loss)
                predict_op = y_pred
            saver = tf.train.Saver()

            i = 0
            batch_size = 20
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                while True:
                    i += 1
                    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trY) + 1, batch_size)):
                        curr_loss, _ = sess.run([loss, train_op], feed_dict={x: trX[start:end], y_true: trY[start:end]})
                    rmse = sess.run(loss, feed_dict={x: teX, y_true: teY})
                    if rmse < 0.02 :
                        print("MSE at convergence", rmse**2)
                        mses[idx] += rmse
                        break
                    if i % 1000 == 0:
                        print("Epoch", i, "RMSE", rmse, "with size of", size)
        mses[idx] = mses[idx]/numExperiments
        idx += 1

    # x = [mses[j][0] for j in range(len(mses))]
    x = layerSize
    y = [mses[j]**2 for j in range(len(mses))]

    plt.plot(x,
             y,
             label="goal")

    plt.xlabel("Hidden Layer Size")
    plt.ylabel("Average MSE at convergence")
    plt.show()

    for size in [8, 50]:
        converged = False
        early_stopped = False
        while True:
            tf.reset_default_graph()
            with tf.variable_scope('Graph') as scope:
                x = tf.placeholder("float", shape=[None, 2], name='inputs')
                y_true = tf.placeholder("float", shape=[None, 1], name='y_true')
                # output of our model
                y_pred = model(x, hidden_dim=size)
                with tf.variable_scope('Loss'):
                    loss = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
                train_op = tf.train.RMSPropOptimizer(learning_rate=0.005, centered=True, momentum=0.1).minimize(loss)
                predict_op = y_pred
            saver = tf.train.Saver()

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                failures = 0
                previous = float("inf")
                i = 0
                batch_size = 20
                rmse_arr = []
                while True:
                    i += 1
                    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trY) + 1, batch_size)):
                        curr_loss, _ = sess.run([loss, train_op], feed_dict={x: trX[start:end], y_true: trY[start:end]})
                    rmse = sess.run(loss, feed_dict={x: teX, y_true: teY})
                    if rmse < 0.02 and converged:
                        break
                    elif rmse < 0.02 and not converged:
                        converged = True
                        print('Converged at {} epochs'.format(i + 1))
                        com = np.vstack((x1.flatten(), y1.flatten())).T
                        predicted = sess.run(predict_op, feed_dict={x: com})
                        predicted = np.reshape(predicted, (-1, 9))

                        values[0] = ([x1, y1, predicted])

                        if trainings[0] == []:
                            trainings[0] = trainings[1]

                        trainings[2] = [
                            0.02 ** 2,
                            np.square(sess.run(loss, feed_dict={x: teX, y_true: teY})),
                            np.square(sess.run(loss, feed_dict={x: vX, y_true: vY})),
                            np.square(sess.run(loss, feed_dict={x: trX, y_true: trY})),
                        ]

                        break

                    if i % 100 == 0:
                        print("Epoch", i, "Failures", failures, "RMSE testing", sess.run(loss, feed_dict={x: teX, y_true: teY}))
                        print("Epoch", i, "Failures", failures, "RMSE training", sess.run(loss, feed_dict={x: trX, y_true: trY}))

                    if not converged and rmse < 0.2:
                        trainings[0] = trainings[1]
                        trainings[1] = [
                            0.02 ** 2,
                            np.square(sess.run(loss, feed_dict={x: teX, y_true: teY})),
                            np.square(sess.run(loss, feed_dict={x: vX, y_true: vY})),
                            np.square(sess.run(loss, feed_dict={x: trX, y_true: trY})),
                        ]

                    if sess.run(loss, feed_dict={x: vX, y_true: vY}) > previous:
                        # if we already have a network with early_stopping, we want to just skip
                        # this step
                        if early_stopped:
                            continue
                        previous = sess.run(loss, feed_dict={x: vX, y_true: vY})
                        failures = failures + 1
                        if failures == 10:
                            print("Early stopping at {}".format(i))
                            early_stopped = True

                            com = np.vstack((x1.flatten(), y1.flatten())).T
                            predicted = sess.run(predict_op, feed_dict={x: com})
                            predicted = np.reshape(predicted, (-1, 9))

                            values[1] = ([x1, y1, predicted])
                            break
                    else:
                        failures = 0
                        previous = sess.run(loss, feed_dict={x: vX, y_true: vY})

            if converged and early_stopped:
                values.append((x1, y1, f(x1, y1)))

                print(trainings)
                plt.plot([0, 1, 2],
                         [trainings[0][0], trainings[1][0], trainings[2][0]],
                         label="goal", color="k")
                plt.plot([0, 1, 2],
                         [trainings[0][1], trainings[1][1], trainings[2][1]],
                         label="test", color="r")
                plt.plot([0, 1, 2],
                         [trainings[0][2], trainings[1][2], trainings[2][2]],
                         label="validation", color="g")
                plt.plot([0, 1, 2],
                         [trainings[0][3], trainings[1][3], trainings[2][3]],
                         label="training", color="b")

                plt.xlabel("2 epochs")
                plt.ylabel("MSE")
                plt.yscale("log")
                plt.legend()
                plt.show()

                fig = plt.figure()
                ax = fig.add_subplot(111)
                a = ax.contour(values[0][0], values[0][1], values[0][2], colors='g')
                h, _ = a.legend_elements()
                a2 = ax.contour(values[1][0], values[1][1], values[1][2], colors='b')
                h1, _ = a2.legend_elements()
                a1 = ax.contour(values[2][0], values[2][1], values[2][2], colors='r')
                h2, _ = a1.legend_elements()

                plt.clabel(a, fontsize=10, colors=plt.cm.Reds(a.norm(a.levels)))
                ax.legend([h[0], h1[0], h2[0]], ['without early stopping ' + str(size) + ' neurons',
                                                 'with early stopping ' + str(size) + ' neurons',
                                                 'target'])
                plt.show()
                break
