import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imrotate
import tensorflow as tf

class HopfieldNetwork(object):
    def hebbian(self):
        self.W = np.zeros([self.num_neurons, self.num_neurons])
        # for image_vector, _ in self.train_dataset:
        #     temp = np.matmul(np.array(image_vector).reshape(784, 1), np.array(image_vector).reshape(1, 784))
        #     self.W = np.add(self.W, temp)
        # np.fill_diagonal(self.W, 0)
        # np.set_printoptions(threshold=np.nan)
        for image_vector, _ in self.train_dataset:
            self.W += np.outer(image_vector, image_vector) / self.num_neurons
        np.fill_diagonal(self.W, 0)

    def storkey(self):
        self.W = np.zeros([self.num_neurons, self.num_neurons])

        for image_vector, _ in self.train_dataset:
            self.W += np.outer(image_vector, image_vector) / self.num_neurons
            net = np.dot(self.W, image_vector)

            pre = np.outer(image_vector, net)
            post = np.outer(net, image_vector)

            self.W -= np.add(pre, post) / self.num_neurons
        np.fill_diagonal(self.W, 0)

    def __init__(self, train_dataset=[], mode='hebbian'):
        self.train_dataset = train_dataset
        self.num_training = len(self.train_dataset)
        self.num_neurons = len(self.train_dataset[0][0])

        if mode == 'hebbian':
            self.hebbian()
        else:
            self.storkey()

    def activate(self, vector):
        changed = True
        while changed:
            changed = False
            indices = [i for i in range(0, len(vector))]

            np.random.shuffle(indices)

            # Vector to contain updated neuron activations on next iteration
            new_vector = np.copy(vector)
            #print("length", new_vector)

            for i in range(0, len(vector)):
                #print("changed")
                neuron_index = indices.pop()

                s = 0
                for pixel_index in range(len(vector)):
                    pixel = vector[pixel_index]
                    if pixel > 0:
                        s += self.W[neuron_index][pixel_index]

                if s > 0:
                    new_vector[neuron_index] = 1
                elif s < 0:
                    new_vector[neuron_index] = -1

                changed = not vector[neuron_index] == new_vector[neuron_index]

            vector = new_vector

        return vector


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = np.concatenate((x_train, x_test)).astype(np.float)
y_train = np.concatenate((y_train, y_test)).astype(np.float)

ones = []
fives = []

for j in range(len(x_train)):
    if y_train[j] == 1:
        ones.append(x_train[j].reshape([1, 784])[0])
    elif y_train[j] == 5:
        fives.append(x_train[j].reshape([1, 784])[0])

ones = [[1 if p > 0 else -1 for p in v] for v in ones]
ones = [(x, 1) for x in ones]
np.random.shuffle(ones)

fives = [[1 if p > 0 else -1 for p in v] for v in fives]
fives = [(x, 5) for x in fives]
np.random.shuffle(fives)

testing_set = ones[500:600] + fives[500:600]
np.random.shuffle(testing_set)

THRESHOLD = 30


# Functions used for testing network classification accuracy

def plot_accuracy(x, y, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)

    plt.xlabel('Training Samples')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.show()


def subshow(img, title='', suptitle=''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.set_title(title)


def show(img, title='', suptitle=''):
    plt.imshow(img)
    plt.title(title)
    plt.suptitle(suptitle)
    plt.show()


def test(network, index, item, ones, fives, sup, plot=False):
    # Measures classification accuracy by diff the activated image vector
    image = np.array(item[0]).reshape(28, 28)

    result = np.array(network.activate(item[0]))

    label = item[1]

    #print("comparing")
    min_distance = float('inf')
    for j in ones:
        #subshow(np.array(j[0]).reshape(28, 28), "Training ones %s" % index, sup)
        dist = np.linalg.norm(result - j[0])
        if dist < min_distance:
            min_distance = dist
            winning_label = j[1]

    #print("comparing fives")
    for j in fives:
        #subshow(np.array(j[0]).reshape(28, 28), "Training 5 %s" % index, sup)
        dist = np.linalg.norm(result - j[0])
        if dist < min_distance:
            min_distance = dist
            winning_label = j[1]

    if plot:
        subshow(image, "Original %s" % index, sup)
        subshow(result.reshape(28, 28), "After %s" % index, sup)
        print("winning label", winning_label)
        plt.show()

    return winning_label == label


    #print("result", result)
    #print(ones[0])
    # if plot:
    #     subshow(image, "Original %s" % index, sup)
    #     subshow(result.reshape(28, 28), "After %s" % index, sup)
    # for j in (np.array(ones) if label == 1 else np.array(fives)):
    #     #print(j)
    #     if plot:
    #         subshow(np.array(j[0]).reshape(28, 28), "Training - Example %s" % index, "stuff")
    #     if np.array_equal(np.multiply(-1, result), np.array(j[0])) or \
    #        np.array_equal(result, np.array(j[0])):
    #         if plot:
    #             print("returning1")
    #             plt.show()
    #         return 1
    #
    # if plot:
    #     print("returning")
    #     plt.show()
    # return 0

    # contrast = np.array(fives[0][0]).reshape(28, 28) if label == 1 else np.array(ones[0][0]).reshape(28, 28)
    #
    # contrast_norm = np.linalg.norm(contrast - result)
    # attempts = [
    #     result,
    #     imrotate(result, angle=30.),
    #     imrotate(result, angle=-30.),
    # ]
    #
    # best_attempt = float('inf')
    #
    # for r in attempts:
    #     for attempt in [r, np.invert(r)]:
    #         attempt_norm = np.linalg.norm(image - attempt)
    #         if attempt_norm < best_attempt:
    #             best_attempt = attempt_norm
    #
    # if plot:
    #     print("guess", best_attempt)
    #     show(image, "Input - Example %s" % index, sup)
    #     show(result, "Output - Example %s" % index, sup)
    #
    # return best_attempt if best_attempt < contrast_norm else float('inf')


# Test Hebbian-based Hopfield network classification accuracy

x = [0 for _ in range(1, 5006)]
y = [0 for _ in range(1, 5006)]

runs = 2
for run in range(runs):
    for i in list([10, 100, 250, 5000]):
        training_set = ones[30:30 + i] + fives[30:30 + i]
        #for image in training_set:
        #    show(np.array(image[0]).reshape(28, 28), "Training", "trained")

        np.random.shuffle(training_set)

        hf_hebbian = hf_hebbian = HopfieldNetwork(
            train_dataset=training_set,
            mode='hebbian'
        )

        hebb_acc = 0.
        for index, image in enumerate(testing_set):
            norm = test(hf_hebbian, index, image, ones[30:30 + i], fives[30:30 + i],
                        "Mode=Hebbian", plot=False)

            #print("norm", norm)
            #if norm <= THRESHOLD:
            if norm == 1:
                hebb_acc += 1

        # x.append(i * 2)
        x[i - 1] += i * 2
        # y.append(hebb_acc / len(testing_set))
        y[i - 1] += hebb_acc / len(testing_set)

        print("hebbian accuracy", hebb_acc, "size", len(testing_set))
        print("Hebbian accuracy trained with {} samples: accuracy {}".format(i * 2, (hebb_acc / len(testing_set))))

print(x)
print(np.divide(x, runs))
plot_accuracy(np.divide(x, runs), np.divide(y, runs), "Hebbian Hopfield Network Accuracy vs. Training Samples Used")

# Test Storkey-based Hopfield network classification accuracy

x = list()
y = list()

for i in range(4, 70):
    training_set_sto = ones[30:30 + i] + fives[30:30 + i]
    np.random.shuffle(training_set_sto)
    hf_storkey = HopfieldNetwork(
        train_dataset=training_set_sto,
        mode='storkey'
    )

    sto_acc = 0.
    #i = 0
    print("starting")
    for index, image in enumerate(testing_set):
        #i += 1
        #if i % 10 == 0:
            #print("i", i)
        # Change to plot=True to see low energy state visualizations
        norm = test(hf_storkey, index, image, ones[30:30 + i], fives[30:30 + i],
                    "Mode=Storkey", plot=False)

        #if norm <= THRESHOLD:
        if norm == 1:
            sto_acc += 1

    x.append(i * 2)
    y.append(sto_acc / len(testing_set))

    # Uncomment below to show computed accuracies
    print("Storkey accuracy trained with {} samples: {}".format(i * 2, (sto_acc / len(testing_set))))

plot_accuracy(x, y, "Storkey Hopfield Network Accuracy vs. Training Samples Used")
