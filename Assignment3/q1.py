import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class network(object):
    def hebbian(self):
        # self.W = np.array([[0, -1, -3, 3],
        #                    [-1, 0, 1, -1],
        #                    [-3, 1, 0, -3],
        #                    [3, -1, -3, 0]])
        self.W = np.zeros([self.num_neurons, self.num_neurons])
        for image_vector, _ in self.train_dataset:
            temp = np.matmul(np.array(image_vector).reshape(len(image_vector), 1),
                             np.array(image_vector).reshape(1, len(image_vector)))
            self.W = np.add(self.W, temp)
        np.fill_diagonal(self.W, 0)

    def storkey(self):
        # https://stats.stackexchange.com/questions/276889/whats-wrong-with-my-algorithm-for-implementing-the-storkey-learning-rule-for-ho
        self.W = np.zeros([self.num_neurons, self.num_neurons])

        for image_vector, _ in self.train_dataset:
            hebbian = np.outer(image_vector, image_vector)
            np.fill_diagonal(hebbian, 0)

            net = np.dot(self.W, image_vector)

            pre = np.outer(image_vector, net)
            post = np.outer(net, image_vector)

            self.W = np.add(self.W, np.divide(np.subtract(hebbian, np.add(pre, post)), self.num_neurons))

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

            new_vector = np.copy(vector)

            for i in range(0, len(vector)):
                neuron_index = indices.pop()

                s = np.dot(new_vector, self.W[neuron_index])

                if s > 0:
                    new_vector[neuron_index] = 1
                elif s < 0:
                    new_vector[neuron_index] = -1

                changed = (not vector[neuron_index] == new_vector[neuron_index]) or changed

            vector = new_vector

        return vector


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype(np.float)
y_train = y_train.astype(np.float)

x_test = x_test.astype(np.float)
y_test = y_test.astype(np.float)

ones = []
fives = []

onesTest = []
fivesTest = []

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

for j in range(len(x_test)):
    if y_test[j] == 1:
        onesTest.append(x_test[j].reshape([1, 784])[0])
    elif y_test[j] == 5:
        fivesTest.append(x_test[j].reshape([1, 784])[0])

onesTest = [[1 if p > 0 else -1 for p in v] for v in onesTest]
onesTest = [(x, 1) for x in onesTest]
np.random.shuffle(onesTest)

fivesTest = [[1 if p > 0 else -1 for p in v] for v in fivesTest]
fivesTest = [(x, 5) for x in fivesTest]
np.random.shuffle(fivesTest)

testing_set = onesTest[0:200] + fivesTest[0:200]
np.random.shuffle(testing_set)


def plot_accuracy(x, y, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)

    plt.xlabel('Training Samples')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.show()


def subshow(img, title=''):
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

    # comparing ones
    min_distance = float('inf')
    for j in ones:
        if plot:
            subshow(np.array(j[0]).reshape(28, 28), "Training ones %s" % index)
        dist = np.linalg.norm(result - j[0])
        if dist < min_distance:
            min_distance = dist
            winning_label = j[1]
        dist = np.linalg.norm(np.multiply(-1, result) - j[0])
        if dist < min_distance:
            min_distance = dist
            winning_label = j[1]

    # comparing fives
    for j in fives:
        if plot:
            subshow(np.array(j[0]).reshape(28, 28), "Training 5 %s" % index)
        dist = np.linalg.norm(result - j[0])
        if dist < min_distance:
            min_distance = dist
            winning_label = j[1]
        dist = np.linalg.norm(np.multiply(-1, result) - j[0])
        if dist < min_distance:
            min_distance = dist
            winning_label = j[1]

    if plot:
        subshow(image, "Original %s" % index, sup)
        subshow(result.reshape(28, 28), "After %s" % index, sup)
        print("winning label", winning_label)
        plt.show()

    return winning_label == label


# hebbian
x = [0 for _ in range(1, 30)]
y = [0 for _ in range(1, 30)]

runs = 5
for run in range(runs):
    print("run", run)
    for i in range(1, 30, 1):
        print("Training on hebbian", i * 2)
        training_set = ones[:i] + fives[:i]

        np.random.shuffle(training_set)

        hf_hebbian = hf_hebbian = network(
            train_dataset=training_set,
            mode='hebbian'
        )

        hebb_acc = 0.
        for index, image in enumerate(testing_set):
            norm = test(hf_hebbian, index, image, ones[:i], fives[:i],
                        "Mode=Hebbian", plot=False)
            hebb_acc += norm

        x[i - 1] += i * 2
        y[i - 1] += hebb_acc / len(testing_set)

        print("hebbian accuracy", hebb_acc, "size", len(testing_set))
        print("Hebbian accuracy trained with {} samples: accuracy {}".format(i * 2, (hebb_acc / len(testing_set))))


plot_accuracy(np.divide(x, runs), np.divide(y, runs), "Accuracy vs training samples")

# storkey

x = [0 for _ in range(1, 30)]
y = [0 for _ in range(1, 30)]

for run in range(runs):
    for i in range(1, 30, 1):
        print("Training on storkey", i * 2)
        training_set_sto = ones[:i] + fives[:i]
        np.random.shuffle(training_set_sto)
        hf_storkey = network(
            train_dataset=training_set_sto,
            mode='storkey'
        )

        ac = 0.
        print("starting")
        for index, image in enumerate(testing_set):
            norm = test(hf_storkey, index, image, ones[:i], fives[:i],
                        "Mode=storkey", plot=False)
            ac += norm

        x[i - 1] += i * 2
        y[i - 1] += ac / len(testing_set)

        print("Storkey accuracy trained with {} samples: {}".format(i * 2, (ac / len(testing_set))))

plot_accuracy(np.divide(x, runs), np.divide(y, runs), "Storkey Accuracy vs training samples")
