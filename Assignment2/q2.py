#Step 1
#We need to define the 31 7x5 element inputs. Each image is going to be a 7x5 2d array and we can simply represent it by
#saying 1= yellow and 0=green. 
#We then want to generate new data (we call this data augmentation) by his specification which is by reversing
#3 bits of the original characters. The idea is we purposefully add noise to data to minimize the generalized error.

#Question to ask: should the 3 reversed bits make sure that it doesn't reverse with another bit that has the identical value
#basically, make sure that the outcome has 3 bits of the original character out of place. (i mean probably yes but just to make sure)
# Another question, how many of these characters do we generate. (1 for each? or more)

#When we make the model, make sure the dimensions are such that the input can take in 35 inputs (7*5=35), and the output layer is a 31 one-hot 
# encoded layer
#  The model should only contain 2 layers feedforward network and we can play with the # of hidden neurons
# use log sigmoidal as our transfer function aka activation function from the hidden layer to output layer

#

