import pickle

import pandas as pd

from MNIST_package import mnist_downloader
from mnist import MNIST
from Framework_NeuralNetwork.libraries import plotting as pt
import numpy as np

from Framework_NeuralNetwork import NeuralNetwork as nn

#Download MNIST Data
# make sure that you don't accidently add the download folder
# to your git (it *should* already be in the .gitignore)
download_folder = "./mnist/"
mnist_downloader.download_and_unzip(download_folder)


# Load MNIST_package Data
mndata = MNIST('mnist', return_type="numpy")
images_train, labels_train = mndata.load_training()
images_validation, labels_validation = mndata.load_testing()

# Instantiate Net
mnist_net = nn.neuralNetwork(activation_function_name="tanh",
                             error_function_name="ce")


# fit and transform Data
mnist_net.scaler.fit(images_train)
images_train_scaled = mnist_net.scaler.transform(images_train)
labels_train_one_hot = np.identity(10, dtype=int)[labels_train]


# Set thetas and Train Net
architecture_layer = [10, 10]
mnist_net.set_thetas(X=images_train_scaled,
                     output_neurons=10,
                     architecture=architecture_layer,
                     )
f1scores_train, error_history, accuracy, a_S = mnist_net.gradient_descent_logistic(X=images_train_scaled,
                                                                                   Y=labels_train_one_hot,
                                                                                   alpha=0.25,
                                                                                   lamb=0,
                                                                                   iterations=100)

# visualize Training results
pt.feature_error_comp(error_histories=error_history,
                      architecture=architecture_layer)

    # Show F1-Score per Digit
x_axis_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
pt.bar_chart(f1_scores=f1scores_train,
             x_axis_labels=x_axis_labels,
             architecture=architecture_layer)

    # Show Confusionmatrix

confusion_matrix = nn.confusion_matrix(a_S, labels_train_one_hot)
pt.plot_confusion(confusion_matrix,
                  label_s="digits",
                  architecture=architecture_layer)


#_____
# Validtion Process

# scale Validation Data
images_validation_scaled = mnist_net.scaler.transform(images_validation)
labels_validation_one_hot = np.identity(len(x_axis_labels), dtype=int)[labels_validation]


# Predict Digits
prediction = mnist_net.predict(images_validation_scaled)


# Visualize results
    # F1Scores
f1scores_val = []
for i in range(len(x_axis_labels)):
    f1_score = nn.f1score(h=prediction,
                          y=labels_validation_one_hot,
                          class_integer=i)
    f1scores_val.append(f1_score)

f1scores_val = pd.DataFrame(f1scores_val)
pt.bar_chart(f1_scores=f1scores_val,
             x_axis_labels=x_axis_labels,
             architecture=architecture_layer)

    # Confusion Matrix
confusion_matrix = nn.confusion_matrix(prediction, labels_validation_one_hot)
pt.plot_confusion(confusion_matrix,
                  label_s="digits",
                  architecture=architecture_layer)



# If Happy save Net as File
pickle.dump(mnist_net, file=open("Hella_nices_mnist_Net", "wb"))

#and use later with:
mnist_net_loaded: nn.neuralNetwork = pickle.load(open("Hella_nices_mnist_Net", "rb"))