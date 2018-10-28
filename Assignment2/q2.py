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



trX, trY = randomizeTestingPoints()
trainingX = trX[0]
trainingY = trY[0]
batchSize = 2

data, labels = get_random_data(trainingX, trainingY, batchSize)

print(len(trainingX))
#For question 2a
recog_error = [] #list of errors for
numExperiments = 5
for size in [5, 10, 15, 20, 25]:
    #we are gonna find the average of 5 runs for each layer size to
    #have a better representation
    recog_error.append([0,0,0,0])
    for i in range(numExperiments):
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
                # epochs = [i for i in range(300)]
                error_rate_list = []
                for i in range(300):
                    for j in range(0, len(data)):
                        curr_loss, _ = sess.run([loss, train_op], feed_dict={x: data[j], y_true: labels[j]})

                    er = sess.run(error_rate, feed_dict={x: trX[0], y_true:trY[0]})
                    error_rate_list.append(er)
                idx = int(size/5-1)
                recog_error[idx][0] += (100*sess.run(error_rate, feed_dict={x: trX[0], y_true:trY[0]}))
                recog_error[idx][1] += (100*sess.run(error_rate, feed_dict={x: trX[1], y_true:trY[0]}))
                recog_error[idx][2] += (100*sess.run(error_rate, feed_dict={x: trX[2], y_true:trY[0]}))
                recog_error[idx][3] += (100*sess.run(error_rate, feed_dict={x: trX[3], y_true:trY[0]})) 
#calculate the average
recog_error_avg = [[errSum/numExperiments  for errSum in errors] for errors in recog_error]
print(recog_error_avg)
###Plot 2a for hidden neurons 5-25
fig, ax = plt.subplots()
ax.plot([i for i in range(4)], recog_error_avg[0], color='r', label='5 Hidden Neurons')
ax.plot([i for i in range(4)], recog_error_avg[1], color='b', label='10 Hidden Neurons')
ax.plot([i for i in range(4)], recog_error_avg[2], color='g', label='15 Hidden Neurons')
ax.plot([i for i in range(4)], recog_error_avg[3], color='y', label='20 Hidden Neurons')
ax.plot([i for i in range(4)], recog_error_avg[4], color='k', label='25 Hidden Neurons')
legend = ax.legend(loc='upper center', shadow=True)
plt.ylabel('Percentage of Recognition Errors')
plt.xlabel('Noise Level')
plt.yscale('linear')
plt.show()

#For question 2b and c
print('Question 2b and c')
for size in [25]:
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
            partC_error_rate_noiseless = [0,0,0,0]
            partC_error_rate_withNoise = [0,0,0,0]
            ######################################################
            
            epochs = [i for i in range(500)]
            error_rate_list = []
            for i in range(500):
                for j in range(0, len(data)):
                    curr_loss, _ = sess.run([loss, train_op], feed_dict={x: data[j], y_true: labels[j]})
                
                er = sess.run(error_rate, feed_dict={x: trX[0], y_true:trY[0]})
                error_rate_list.append(er)
                
                # print(er)
            #try 5 different testing points
            for i in range(5):
                trXc, trYc = randomizeTestingPoints() 
                partC_error_rate_noiseless[0] += (100*sess.run(error_rate, feed_dict={x: trXc[0], y_true:trY[0]}))
                partC_error_rate_noiseless[1] += (100*sess.run(error_rate, feed_dict={x: trXc[1], y_true:trY[0]}))
                partC_error_rate_noiseless[2] += (100*sess.run(error_rate, feed_dict={x: trXc[2], y_true:trY[0]}))
                partC_error_rate_noiseless[3] += (100*sess.run(error_rate, feed_dict={x: trXc[3], y_true:trY[0]}))

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
            #shuffle the data
            random_idx = np.random.permutation(len(noisy_data))
            random_noisy_data = []
            random_noisy_label = []
            for i in random_idx:
                random_noisy_data.append(noisy_data[i])
                random_noisy_label.append(noisy_labels[i])

            error_rate_list = []
            #train until performance goal for noisy data is 0.01 but we will be using 0.02 because the smallest value 
            #the error can be where it is not 0 is 0.0107 which is greater than 0.01
            counter = 0
            print('Training on noisy data')
            for i in range(100):
                for j in range(0, len(noisy_data)):
                    curr_loss, _ = sess.run([loss, train_op], feed_dict={x:random_noisy_data[j], y_true: random_noisy_label[j]})
                er1 = sess.run(error_rate, feed_dict={x:trX[1], y_true:trY[0]})    
                er2 = sess.run(error_rate, feed_dict={x:trX[2], y_true:trY[0]})   
                er3 = sess.run(error_rate, feed_dict={x:trX[3], y_true:trY[0]})   
                er = (er1 + er2 + er3)/3.0
                counter += 1
                if er <= 0.02:
                    break
            print('Done... training on ideal data again')
            #now retrain on on ideal set again until the error is 0 or less than 0.01
            for i in range(300):
                for j in range(0, len(data)):
                    curr_loss, _ = sess.run([loss, train_op], feed_dict={x: data[j], y_true: labels[j]})
                
                er = sess.run(error_rate, feed_dict={x: trX[0], y_true:trY[0]})
                print(er)
                error_rate_list.append(er)
                if er < 0.02:
                    break
                

            # print(len(error_rate_list), error_rate_list)
            epochs = [i for i in range(len(error_rate_list))]
            plt.plot(epochs,error_rate_list)
            plt.ylabel('Training Error')
            plt.xlabel('Epochs')
            plt.yscale('log')
            try:
                plt.show()
            except:
                print("didn't work")

            #part c: create testing points
            fig, ax = plt.subplots()
            #try 5 different testing points 
            for i in range(5):
                randomizeTestingPoints()
                partC_error_rate_withNoise[0] += (100*sess.run(error_rate, feed_dict={x: trX[0], y_true:trY[0]})) 
                partC_error_rate_withNoise[1] += (100*sess.run(error_rate, feed_dict={x: trX[1], y_true:trY[1]})) 
                partC_error_rate_withNoise[2] += (100*sess.run(error_rate, feed_dict={x: trX[2], y_true:trY[2]})) 
                partC_error_rate_withNoise[3] += (100*sess.run(error_rate, feed_dict={x: trX[3], y_true:trY[3]})) 
            ax.plot([i for i in range(4)], partC_error_rate_withNoise, color='r', label='trained with noise')
            ax.plot([i for i in range(4)], partC_error_rate_noiseless, color='b', label='trained without noise')
            legend = ax.legend(loc='upper center', shadow=True)

            plt.ylabel('Percentage of Recognition Errors')
            plt.xlabel('Noise Level')
            plt.yscale('linear')
            plt.show()



            





