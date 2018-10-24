import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt



# Function approximator model
def model(x, hidden_dim=8):
    input_dim = 35
    output_dim = 31
    stdev = 0.01
    with tf.variable_scope('FunctionApproximator'):
        w_h1 = tf.get_variable('w_h1', shape=[input_dim, hidden_dim],
                               initializer=tf.random_normal_initializer(stddev=stdev))
        b_h1 = tf.get_variable('b_h1', shape=[hidden_dim], initializer=tf.constant_initializer(1.))

        z = tf.nn.sigmoid(tf.matmul(x, w_h1) + b_h1)

        w_o = tf.get_variable('w_o', shape=[hidden_dim, output_dim],
                              initializer=tf.random_normal_initializer(stddev=stdev))

    return tf.matmul(z, w_o)



lines = open("./q2-patterns.txt").read().splitlines()


def randomizeTestingPoints():
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
    return trX, trY

trX, trY = randomizeTestingPoints()

# training on images with nosie of 0 and 3
# can do np.vstack((trX[0], trX[1], trX[2], trX[3])) to train on all noises
#trainingX = np.vstack((trX[0], trX[3]))
#trainingY = np.vstack((trY[0], trY[3]))
trainingX = trX[0]
trainingY = trY[0]
idx = np.random.permutation(len(trainingX))
# idx = [i for i in idx] * 10
np.random.shuffle(idx)
# print('INDEX', idx)
batchSize = 2

def get_random_data(x_data, y_data, batch_size=2):
    idx = np.random.permutation(len(x_data))
    data = []
    labels = []
    for j in range(0, len(idx), batchSize):
        data.append([])
        labels.append([])
        for k in range(0, batchSize):
            if j + k >= len(x_data):
                break
            data[int(j/batchSize)].append(x_data[idx[j + k]])
            labels[int(j/batchSize)].append(y_data[idx[(j + k)]])
    return data, labels   

data, labels = get_random_data(trainingX, trainingY, batchSize)
question = 'b'

# for j in range(0, len(idx), batchSize):
#     data.append([])
#     labels.append([])
#     for k in range(0, batchSize):
#         if j + k >= len(trainingX):
#             break
#         print(j + k)
#         data[int(j/batchSize)].append(trainingX[idx[j + k]])
#         labels[int(j/batchSize)].append(trainingY[idx[j + k]])

print(len(trainingX))
#For question 2a
recog_error = [] #list of errors for 
for size in [5, 10, 15, 20, 25]:
    recog_error.append([])
    tf.reset_default_graph()
    print("Training with {} number of hidden neurons".format(size))
    with tf.variable_scope('Graph') as scope:
        x = tf.placeholder("float", shape=[None, 35], name='inputs')
        y_true = tf.placeholder("float", shape=[None, 31], name='y_true')
        y_true_cls = tf.argmax(y_true, 1)
        # output of our model
        y_pred = model(x, hidden_dim=size)
        with tf.variable_scope('Loss'):
            cost = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
            loss = tf.reduce_mean(cost)  # compute costs
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
        predict_op = tf.argmax(y_pred, 1)
        correct_prediction = tf.equal(predict_op, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        error_rate = 1 - accuracy

    with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ######################################################
            epochs = [i for i in range(500)]
            error_rate_list = []
            for i in range(500):
                for j in range(0, len(data)):
                    curr_loss, _ = sess.run([loss, train_op], feed_dict={x: data[j], y_true: labels[j]})
                
                er = sess.run(error_rate, feed_dict={x: trX[0], y_true:trY[0]})
                error_rate_list.append(er)
            idx = int(size/5-1)
            recog_error[idx].append(100*sess.run(error_rate, feed_dict={x: trX[0], y_true:trY[0]}))
            recog_error[idx].append(100*sess.run(error_rate, feed_dict={x: trX[1], y_true:trY[0]}))
            recog_error[idx].append(100*sess.run(error_rate, feed_dict={x: trX[2], y_true:trY[0]}))
            recog_error[idx].append(100*sess.run(error_rate, feed_dict={x: trX[3], y_true:trY[0]}))
###Plot 2a for hidden neurons 5-25
fig, ax = plt.subplots()
ax.plot([i for i in range(4)], recog_error[0], color='r', label='5 Hidden Neurons')
ax.plot([i for i in range(4)], recog_error[1], color='b', label='10 Hidden Neurons')
ax.plot([i for i in range(4)], recog_error[2], color='g', label='15 Hidden Neurons')
ax.plot([i for i in range(4)], recog_error[3], color='y', label='20 Hidden Neurons')
ax.plot([i for i in range(4)], recog_error[4], color='k', label='25 Hidden Neurons')
legend = ax.legend(loc='upper center', shadow=True)
plt.ylabel('Percentage of Recognition Errors')
plt.xlabel('Noise Level')
plt.yscale('linear')
plt.show()

#For question 2b and c
for size in [15]:
    tf.reset_default_graph()
    print("Training with {} number of hidden neurons".format(size))
    with tf.variable_scope('Graph') as scope:
        x = tf.placeholder("float", shape=[None, 35], name='inputs')
        y_true = tf.placeholder("float", shape=[None, 31], name='y_true')
        y_true_cls = tf.argmax(y_true, 1)
        # output of our model
        y_pred = model(x, hidden_dim=size)
        with tf.variable_scope('Loss'):
            cost = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
            loss = tf.reduce_mean(cost)  # compute costs
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
        predict_op = tf.argmax(y_pred, 1)
        correct_prediction = tf.equal(predict_op, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        error_rate = 1 - accuracy

        saver = tf.train.Saver()
        #Showing fig 13a
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            #this data will be used for testing and graphing part C
            partC_error_rate_noiseless = []
            partC_error_rate_withNoise = []
            trXc, trYc = randomizeTestingPoints() 
            ######################################################
            
            epochs = [i for i in range(500)]
            error_rate_list = []
            for i in range(500):
                for j in range(0, len(data)):
                    curr_loss, _ = sess.run([loss, train_op], feed_dict={x: data[j], y_true: labels[j]})
                
                er = sess.run(error_rate, feed_dict={x: trX[0], y_true:trY[0]})
                error_rate_list.append(er)
                
                # print(er)
            partC_error_rate_noiseless.append(100*sess.run(error_rate, feed_dict={x: trXc[0], y_true:trY[0]}))
            partC_error_rate_noiseless.append(100*sess.run(error_rate, feed_dict={x: trXc[1], y_true:trY[0]}))
            partC_error_rate_noiseless.append(100*sess.run(error_rate, feed_dict={x: trXc[2], y_true:trY[0]}))
            partC_error_rate_noiseless.append(100*sess.run(error_rate, feed_dict={x: trXc[3], y_true:trY[0]}))

            plt.plot(epochs,error_rate_list)
            plt.ylabel('Training Error')
            plt.xlabel('Epochs')
            plt.yscale('log')
            plt.show()

            #Now train for 10 passes on noisy data 
            noisy_data1, noisy_labels1 = get_random_data(trX[1], trY[0], batchSize)
            noisy_data2, noisy_labels2 = get_random_data(trX[2], trY[0], batchSize)
            noisy_data3, noisy_labels3 = get_random_data(trX[3], trY[0], batchSize)
            
            noisy_data = noisy_data1+noisy_data2+noisy_data3
            noisy_labels = noisy_labels1+noisy_labels2+noisy_labels3
            
            all_data = trX[0]+trX[1]+trX[2]+trX[3]
            all_labels = trY[0]+trY[0]+trY[0]+trY[0]
            error_rate_list = []
            for i in range(10):
                for j in range(0, len(noisy_data)):
                    curr_loss, _ = sess.run([loss, train_op], feed_dict={x:noisy_data[j], y_true: noisy_labels[j]})
                # print(x)
                # print(y_true)
                er0 = sess.run(error_rate, feed_dict={x: trX[0], y_true:trY[1]})
                er1 = sess.run(error_rate, feed_dict={x: trX[1], y_true:trY[1]})
                er2 = sess.run(error_rate, feed_dict={x: trX[2], y_true:trY[1]})
                er3 = sess.run(error_rate, feed_dict={x: trX[3], y_true:trY[1]})
                
                er = (er0+er1+er2+er3)/4
                error_rate_list.append(er)

            #now retrain on on ideal set again
            for i in range(300):
                for j in range(0, len(data)):
                    curr_loss, _ = sess.run([loss, train_op], feed_dict={x: data[j], y_true: labels[j]})
                
                er0 = sess.run(error_rate, feed_dict={x: trX[0], y_true:trY[1]})
                er1 = sess.run(error_rate, feed_dict={x: trX[1], y_true:trY[1]})
                er2 = sess.run(error_rate, feed_dict={x: trX[2], y_true:trY[1]})
                er3 = sess.run(error_rate, feed_dict={x: trX[3], y_true:trY[1]})
                
                er = (er0+er1+er2+er3)/4
                error_rate_list.append(er)

            # print(len(error_rate_list), error_rate_list)
            epochs = [i for i in range(len(error_rate_list))]
            plt.plot(epochs,error_rate_list)
            plt.ylabel('Training Error')
            plt.xlabel('Epochs')
            plt.yscale('log')
            plt.show()

            #part c: create testing points
            fig, ax = plt.subplots()
            randomizeTestingPoints()
            partC_error_rate_withNoise.append(100*sess.run(error_rate, feed_dict={x: trX[0], y_true:trY[0]})) 
            partC_error_rate_withNoise.append(100*sess.run(error_rate, feed_dict={x: trX[1], y_true:trY[1]})) 
            partC_error_rate_withNoise.append(100*sess.run(error_rate, feed_dict={x: trX[2], y_true:trY[2]})) 
            partC_error_rate_withNoise.append(100*sess.run(error_rate, feed_dict={x: trX[3], y_true:trY[3]})) 
            ax.plot([i for i in range(4)], partC_error_rate_withNoise, color='r', label='trained with noise')
            ax.plot([i for i in range(4)], partC_error_rate_noiseless, color='b', label='trained without noise')
            legend = ax.legend(loc='upper center', shadow=True)

            plt.ylabel('Percentage of Recognition Errors')
            plt.xlabel('Noise Level')
            plt.yscale('linear')
            plt.show()



            





