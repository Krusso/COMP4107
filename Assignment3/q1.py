import argparse
import tensorflow as tf 
import sys
from sklearn import datasets
import numpy as np

class MnistDataset:
    def __init__(self):
        digits = datasets.load_digits()
        self.data = digits.data #shape(1797, 64)
        self.label = digits.target #shape(1797,)
        self.size = len(self.data)
    
    def subsample(self, *numbers):
        """
            input: list of numbers that we want to subsample
            returns: a tuple of (data, label) where the data/label only contains the specified numbers as labels
        """
        num = np.asarray(numbers)
        subsample_data = []
        subsample_label = []
        for i in range(self.size):
            if self.label[i] in num:
                subsample_data.append(self.data[i])
                subsample_label.append(self.label[i])
        return subsample_data, subsample_label

def hopfield(input):
    #Todo
    print("Todo: make hopfield")
    
    return None

#Main method:
def main(unused_argv):

    mnist = MnistDataset()
    ss_data, ss_label = mnist.subsample(1,5)

    x = tf.placeholder("float", shape=[None, 64], name='inputs')
    y_true = tf.placeholder("float", shape=[None, 10], name='label')

    y_pred = hopfield(x)

    with tf.variable_scope('Cost'):
        #Todo
        print('todo: write the cost/loss function here')

    print("todo: write the optimizer_op, correct_predition, accuracy, error_rate ")
    #Todo

    saver = tf.train.Saver()


    #Now we train the network here:





#We run main method here:

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #Add arguments here if you'd like

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    