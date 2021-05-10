from os import listdir
from Framework_NeuralNetwork.libraries import logistic_regression as lr, feature_scaling as fs
import numpy as np
import pandas as pd
from tqdm import tqdm


def results_to_csv(h, labels_dictionary):
    out_csv = pd.DataFrame()
    predictions = h.argmax(axis=1)
    out_csv['event'] = np.array([labels_dictionary[item] for item in predictions])
    return out_csv


# returns f1Score of specified Integer(gesture or...)
def f1score(h, y, class_integer):
    h = h.copy()
    for i in range(len(h)):
        if h[i].max() < 1:
            h[i][h[i].argmax()] = 1
    true_positives = (h[:, class_integer] == 1) & (y.argmax(axis=1) == class_integer)
    false_positives = (h[:, class_integer] == 1) & (y.argmax(axis=1) != class_integer)
    false_negatives = (h[:, class_integer] != 1) & (y.argmax(axis=1) == class_integer)

    if np.sum(true_positives) == 0 and np.sum(false_positives == 0):
        return -0.01
    if (np.sum(true_positives) + np.sum(false_negatives)) == 0:
        return -0.01

    precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))
    recall = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))

    f1 = (2 * (precision * recall)) / ((precision + recall) + 0.0000001)
    return f1


#returns confusion Matrix out of Prediction and ground truth
def confusion_matrix(H, Y):
    confusion_m = np.zeros((Y.shape[1], Y.shape[1]))

    h_sum = np.sum(H.argmax(axis=1), axis=0)
    y_sum = np.sum(Y, axis=0)
    for i in range(Y.shape[1]):
        current_class_predictions = H[Y.argmax(axis=1) == i]

        for j in range(Y.shape[1]):
            current_count = np.count_nonzero(
                current_class_predictions.argmax(axis=1) == j)  # Wie viele I's wurden als J's identifiziert
            confusion_m[i, j] = current_count

    conf_sum = np.sum(confusion_m, axis=1)
    return confusion_m

#returns Accuracy per predicted class
def class_accuracy(confusion_matrix):
    per_class_accuracy = np.diag(confusion_matrix) / (confusion_matrix.sum(axis=1) + 1)
    per_class_accuracy = np.nan_to_num(per_class_accuracy, nan=0)
    output = (per_class_accuracy).round(2)
    return output



def get_ground_truth(X, gestures_dic):
    Y_validation = X['ground_truth']

    for key in gestures_dic:
        mask = Y_validation == key
        Y_validation[mask] = gestures_dic[key]

    Y_validation = np.array(Y_validation).astype(int)
    Y_validation_one_hot = np.identity(len(gestures_dic), dtype=int)[Y_validation]
    return Y_validation_one_hot


def cutOff(prediction, cut_off_array):
    for i in range(prediction.shape[0]):
        for j in range(1, len(cut_off_array)+1):
            if prediction[i].argmax(axis=0) == j:
                if prediction[i].max(axis=0) < cut_off_array[j-1]:
                    prediction[i][0] = 1
    return prediction


def cluster(prediction_cut, range_int):
    prediction_cut = prediction_cut.copy()
    prediction_clustered = np.zeros(len(prediction_cut))
    for i in range(0, prediction_cut.shape[0], range_int):

        if prediction_cut.shape[0] - i >= range_int:
            argmax = prediction_cut.argmax(axis=1)[i: i + range_int]
            most_frequent = np.argmax(np.bincount(argmax[0: range_int - 1]))
            for k in range(i, i + range_int):
                prediction_clustered[k] = most_frequent

        else:  # if an odd number remains
            last = prediction_cut.shape[0] - i
            argmax = prediction_cut.argmax(axis=1)[i:i + last]
            if last == 1:
                most_frequent = argmax[0]
            else:
                most_frequent = np.argmax(np.bincount(argmax[0:last - 1]))

            for k in range(i, i + last):
                prediction_clustered[k] = most_frequent

    return prediction_clustered


class neuralNetwork():
    def __init__(self, activation_function_name, error_function_name):  # architetcutre = [5,10,5]

        self.activation_function_name = activation_function_name
        self.error_function_name = error_function_name
        self.ACTIVATION_FUNCTIONS = {
            "sigmoid": lr.sigmoid,
            "tanh": lr.tanh,
        }
        self.DERIVATIVE_FUNCTIONS = {
            "sigmoid": lr.der_sigmoid,
            "tanh": lr.der_tanh,
        }
        self.ERROR_FUNCTIONS = {
            "mse": lr.mean_squared_error,
            "ce": lr.cross_entropy
        }

        self.thetas = []
        self.biases = []
        self.scaler = fs.StandardScaler()

    def set_thetas(self, X, output_neurons, architecture=[0]):
        if architecture == [0]:
            self.thetas.append(np.random.normal(0.0, pow(X.shape[1], -0.5), (1, X.shape[1])))
            self.biases.append(np.random.normal(0.0, pow(X.shape[1], -0.5), (1, 1)))
        else:
            self.thetas.append(np.random.normal(0.0, pow(X.shape[1], -0.5), (architecture[0], X.shape[1])))  # Input
            self.biases.append(np.random.normal(0.0, pow(X.shape[1], -0.5), (architecture[0], 1)))
            for i in range(len(architecture) - 1):  # hiddenLayers
                self.thetas.append(
                    np.random.normal(0.0, pow(architecture[i], -0.5), (architecture[i + 1], architecture[i])))
                self.biases.append(np.random.normal(0.0, pow(architecture[i], -0.5), (architecture[i + 1], 1)))
            self.thetas.append(
                np.random.normal(0.0, pow(architecture[-1], -0.5), (output_neurons, architecture[-1])))  # Output
            self.biases.append(np.random.normal(0.0, pow(architecture[-1], -0.5), (output_neurons, 1)))

    def errorF(self, h, y):
        return self.ERROR_FUNCTIONS[self.error_function_name](h, y)

    def activationF(self, X):
        return self.ACTIVATION_FUNCTIONS[self.activation_function_name](X)

    def derivate(self, X):
        return self.DERIVATIVE_FUNCTIONS[self.activation_function_name](X)

    def predict(self, X):
        a = [None] * len(self.thetas)
        iterations_j = len(a)
        a[0] = self.activationF(X @ self.thetas[0].T + self.biases[0].T)
        for j in range(1, iterations_j - 1):
            a_left = a[j - 1]
            thetas_right = self.thetas[j].T
            bias_right = self.biases[j].T
            a[j] = self.activationF(a_left @ thetas_right + bias_right)

        a[iterations_j - 1] = (a[iterations_j - 2] @ self.thetas[iterations_j - 1].T) + self.biases[iterations_j - 1].T

        # Softmax + Error
        a_S = softmax(a[-1])
        return np.array(a_S)



    def gradient_descent_linear(self, X,Y, alpha, iterations):
        error_h =[]
        samplesize = X.shape[0]
        for i in range(iterations):
            h = self.activationF(X @ self.thetas[0].T + self.biases[0])
            error = self.errorF(h, Y)
            error_h.append(error)
            gradients = (1/samplesize) * (h-Y) * self.thetas[0]
            self.thetas[0] -= alpha * gradients
        return error_h

    def gradient_descent(self, X, Y, alpha, lamb, iterations):
        if self.error_function_name == "mse":
            return self.gradient_descent_linear(X, Y, alpha, iterations)
        if self.error_function_name == "ce":
            return self.gradient_descent_logistic(X, Y, alpha, lamb, iterations)
        return "no valid Error Function"

    def gradient_descent_logistic(self, X, Y, alpha, lamb, iterations):

        samplesize = len(X)
        error_history = []
        f1scores = []

        a = [None] * len(self.thetas)

        for i in tqdm(range(iterations)):

            # Forward Propagation
            iterations_j = len(a)
            a[0] = self.activationF(X @ self.thetas[0].T + self.biases[0].T)
            for j in range(1, iterations_j - 1):
                a_left = a[j - 1]
                thetas_right = self.thetas[j].T
                bias_right = self.biases[j].T
                a[j] = self.activationF(a_left @ thetas_right + bias_right)

            a[iterations_j - 1] = (a[iterations_j - 2] @ self.thetas[iterations_j - 1].T) + self.biases[iterations_j - 1].T

            # Softmax + Error
            a_S = softmax(a[-1])
            J = self.errorF(a_S, Y) - ((lamb / (2 * X.shape[0])) * np.sum(self.thetas[-1] ** 2))  # Regularization
            error_history.append(J)

            error_length = len(a)
            error = [None] * error_length
            delta_thetas = [None] * error_length

            assert len(error) == error_length

            iterations_j = len(error)

            if i == iterations - 1:
                continue

            # Rightes Layer
            error[-1] = a_S - Y
            derivate_1 = self.derivate(a[-1])
            a_left = a[len(a) - 2]
            delta_thetas[-1] = ((error[-1]).T @ a_left) / samplesize

            for j in reversed(range(1, iterations_j - 1)):
                error_right = error[j + 1]
                thetas_right = self.thetas[j + 1]
                derivate_j = self.derivate(a[j])
                error_hidden = error_right @ thetas_right * derivate_j

                error[j] = error_hidden
                a_left = a[j - 1]
                delta_thetas[j] = (error_hidden.T @ a_left) / samplesize

            # Leftest Layer
            error[0] = (error[1] @ self.thetas[1]) * self.derivate(a[0])
            delta_thetas[0] = (error[0].T @ X) / samplesize

            iterations_k = len(a)
            for k in range(iterations_k):
                self.biases[k] = self.biases[k] - alpha * (error[k].mean() + ((lamb / samplesize) * self.biases[k]))
                self.thetas[k] = self.thetas[k] - alpha * (delta_thetas[k] + ((lamb / samplesize) * self.thetas[k]))

            accuracy = (a_S.argmax(axis=1) == Y.argmax(axis=1)).mean()
            print("accuracy")
            print(accuracy)
            print("____________")

        for i in range(Y.shape[1]):
            f1 = f1score(a_S, Y, i)
            f1scores.append(f1)

        f1scores = pd.DataFrame(f1scores)
        return f1scores, error_history, accuracy, a_S




def categorical_cross_entropy(h, y_one_hot):
    h = np.clip(h, a_min=0.000000001, a_max=None)
    entropy = y_one_hot * np.log(h)
    entropy_sum = entropy.sum(axis=1)
    J = np.mean(entropy_sum)

    return -J


def softmax(o):
    e = (np.e ** o).sum(axis=1)[:, np.newaxis]

    return (np.e ** o) / e