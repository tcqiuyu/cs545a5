__author__ = 'Qiu'

import numpy as np
import matplotlib.pyplot as plt
import time, datetime
import math


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x) ** 2


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        """
        layers: A list containing the number of units in each layer.
               Should contain at least two values
        activation: The activation function to be used. Can be
               "logistic" or "tanh"
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        self.num_layers = len(layers) - 1
        self.weights = [np.random.randn(layers[i - 1] + 1, layers[i] + 1) / 10 for i in range(1, len(layers) - 1)]
        self.weights.append(np.random.randn(layers[-2] + 1, layers[-1]) / 10)

    def forward(self, x, linear=False):
        """
        compute the activation of each layer in the network
        """
        a = [x]
        for i in range(self.num_layers):
            ai = self.activation(np.dot(a[i], self.weights[i]))
            if i == self.num_layers - 1 and linear == True:
                ai = np.dot(a[i], self.weights[i])
            if (len(a) < self.num_layers):
                ai[-1] = 1
            a.append(ai)
        return a

    def backward(self, y, a, linear=False):
        """
        compute the deltas for example i
        """

        if linear:
            deltas = [(y - a[-1])] * 1
        else:
            deltas = [(y - a[-1]) * self.activation_deriv(a[-1])]

        for l in range(len(a) - 2, 0, -1):  # we need to begin at the second to last layer
            deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
        deltas.reverse()
        return deltas

    def fit(self, X, y, learning_rate=0.2, epochs=50, _lambda=0.0, linear=False):
        X = np.asarray(X)
        temp = np.ones((X.shape[0], X.shape[1] + 1))
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.asarray(y)

        for k in range(epochs):
            # if k % 10 == 0: print "***************** ", k, "epochs  ***************"
            I = np.random.permutation(X.shape[0])
            for i in I:
                a = self.forward(X[i], linear)
                deltas = self.backward(y[i], a, linear)
                # update the weights using the activations and deltas:
                for i in range(len(self.weights)):
                    layer = np.atleast_2d(a[i])
                    delta = np.atleast_2d(deltas[i])
                    N = len(a[-1])
                    self.weights[i] += learning_rate * (layer.T.dot(delta) + _lambda / N * self.weights[i])

    def predict(self, x):
        x = np.asarray(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a


def test_digits_single_hidden_layer(hidden_layer_units, _lambda=0.0, linear=False):
    from sklearn.cross_validation import train_test_split
    from sklearn.datasets import load_digits
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.preprocessing import LabelBinarizer

    digits = load_digits()
    X = digits.data
    y = digits.target
    X /= X.max()  # Normalize data

    nn = NeuralNetwork([64, hidden_layer_units, 10], 'logistic')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    labels_train = LabelBinarizer().fit_transform(y_train)
    labels_test = LabelBinarizer().fit_transform(y_test)
    nn.fit(X_train, labels_train, epochs=100, _lambda=_lambda, linear=linear)
    predictions = []
    for i in range(X_test.shape[0]):
        o = nn.predict(X_test[i])
        predictions.append(np.argmax(o))
    cm = confusion_matrix(y_test, predictions)
    correct_prediction = 0.0
    for i in range(cm.shape[0]):
        correct_prediction += cm[i][i]
    # print classification_report(y_test, predictions)
    print "***************** (" + time.strftime(
        "%X") + ") Hidden Layer Units = " + hidden_layer_units.__str__() + " *****************"
    accuracy = correct_prediction / X_test.shape[0]
    print "accuracy = " + accuracy.__str__()
    return accuracy


def test_digits_two_hidden_layers(hidden_layer1_units, hidden_layer2_units):
    from sklearn.cross_validation import train_test_split
    from sklearn.datasets import load_digits
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.preprocessing import LabelBinarizer

    digits = load_digits()
    X = digits.data
    y = digits.target
    X /= X.max()  # Normalize data

    nn = NeuralNetwork([64, hidden_layer1_units, hidden_layer2_units, 10], 'logistic')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    labels_train = LabelBinarizer().fit_transform(y_train)
    labels_test = LabelBinarizer().fit_transform(y_test)
    nn.fit(X_train, labels_train, epochs=100)
    predictions = []
    for i in range(X_test.shape[0]):
        o = nn.predict(X_test[i])
        predictions.append(np.argmax(o))
    cm = confusion_matrix(y_test, predictions)
    correct_prediction = 0.0
    for i in range(cm.shape[0]):
        correct_prediction += cm[i][i]
    # print classification_report(y_test, predictions)
    print "***************** (" + time.strftime("%X") + ") Hidden Layer Units = " \
          + hidden_layer1_units.__str__() + ", " + hidden_layer2_units.__str__() + " *****************"
    accuracy = correct_prediction / X_test.shape[0]
    print accuracy
    return accuracy

if __name__ == '__main__':
    max_hidden_layer_units_1 = 100
    ## single hidden layer
    unit_1 = 1
    units_1 = []
    accuracy_1 = []
    while unit_1 <= max_hidden_layer_units_1:
        units_1.append(unit_1)
        unit_1 += math.ceil(unit_1 / 10)
    units_1 = units_1[1:]
    print units_1
    for unit in units_1:
        accuracy_1.append(test_digits_single_hidden_layer(unit))

    plt_units = np.asarray(units_1)
    plt_accuracy = np.asarray(accuracy_1)
    plt.plot(plt_units, plt_accuracy, "b")
    plt.xlabel("Hidden Layer Units")
    plt.ylabel("Accuracy")
    plt.show()

    ## two hidden layers
    start_time = time.time()
    max_hidden_layer_units_2 = 128
    unit_2 = 1
    units_2 = []
    accuracy_2 = []
    while unit_2 <= max_hidden_layer_units_2:
        units_2.append(unit_2)
        unit_2 += math.ceil(unit_2 / 10)
        unit_2 += 1
    print units_2
    units_2 = units_2[1:]
    accuracy_mat_2 = np.zeros((len(units_2), len(units_2)))

    for i in range(len(units_2)):
        for j in range(len(units_2)):
            accuracy_mat_2[i][j] = test_digits_two_hidden_layers(units_2[i], units_2[j])

    end_time = time.time()
    elapse = int(end_time - start_time)
    print "***************** Finished! Time elapsed: " + \
          datetime.timedelta(seconds=elapse).__str__() + " *****************"

    plt_units_1 = np.asarray(units_2)
    plt_units_2 = np.asarray(units_2)
    plt.pcolor(plt_units_1, plt_units_2, accuracy_mat_2, cmap='gnuplot')
    plt.xlabel("Hidden Layer 1 Units")
    plt.ylabel("Hidden Layer 2 Units")
    plt.colorbar()
    plt.show()

    # linear
    max_hidden_layer_units_3 = 200
    unit_3 = 1
    units_3 = []
    accuracy_3 = []
    while unit_3 <= max_hidden_layer_units_3:
        units_3.append(unit_3)
        unit_3 += math.ceil(unit_3 / 10)
    units_3 = units_3[1:]
    print units_3
    for unit in units_3:
        accuracy_3.append(test_digits_single_hidden_layer(unit, linear=True))
    plt_units = np.asarray(units_3)
    plt_accuracy = np.asarray(accuracy_3)
    plt.plot(plt_units, plt_accuracy, "b")
    plt.xlabel("Hidden Layer Units")
    plt.ylabel("Accuracy")
    plt.show()