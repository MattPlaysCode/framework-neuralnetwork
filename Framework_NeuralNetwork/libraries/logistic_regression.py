import numpy as np
import matplotlib.pyplot as plt


def tanh(z):
    return np.tanh(z)


def der_tanh(z):
    return 1 - z ** 2


def sigmoid(z):
    return 1 / (1 + np.e ** -z)


def der_sigmoid(z):
    return z * (1 - z)


def relu(z):
    return np.maximum(0, z)


def der_relu(z):
    return (z > 0).astype(float)


def linear_regression(X, thetas):
    return X @ thetas


def logistic_regression(X, thetas):
    return sigmoid(linear_regression(X, thetas))


def mean_squared_error(h, y):
    return np.mean((h - y) ** 2)


def cross_entropy(h, y):
    h = np.clip(h, 0.000000001, 0.99999999)
    return np.mean(- y * np.log(h) - (1 - y) * np.log(1 - h))


def logistic_regression_derivative(h, y, X):
    m = len(X)
    return (1 / m) * ((h - y) @ X)


def gradient_descent_logistic_regression(X, y, initial_thetas, alpha, iterations):
    trained_thetas = initial_thetas.copy()

    error_history = []

    for i in range(iterations):
        h = logistic_regression(X, trained_thetas)
        error = cross_entropy(h, y)
        error_history.append(error)

        trained_thetas -= alpha * logistic_regression_derivative(h, y, X)

    return trained_thetas, error_history


def visualize_error_history(error_history):
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(error_history, label="Cross Entropy")

    # upper_ylim = np.quantile(error_history, 0.99)
    # ax.set_ylim(-0.001, upper_ylim)

    ax.set_xlabel("iterations")
    ax.set_ylabel("Cross Entropy")
    ax.set_title("Learning Curve")
    fig.legend()


def visualize_f1_score(f1array):
    fig, ax = plt.subplots(len(f1array), figsize=(14, 7))

    for i in range(len(f1array)):
        ax[i].plot(f1array[i], label="F1Score")
        ax[i].set_xlabel("iterations")
        ax[i].set_ylabel("F1")
        ax[i].set_title("Learning Curve")

    # upper_ylim = np.quantile(error_history, 0.99)
    # ax.set_ylim(-0.001, upper_ylim)

    fig.legend()


def accuracy(h, y):
    return np.mean(h.round() == y)
