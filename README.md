# Background
* This framework was created for a gesture recognition project in the Human Computer Interaction course at the University of Würzburg.
* You can see the result **[here](https://www.youtube.com/watch?v=LjNq6iJn_EQ)**.
* The development was done in collaboration with my colleague **[Jan](https://github.com/schmifers)**.

# Prerequesites
* you need at least Python 3.7
* you need to have **[pipenv](https://pypi.org/project/pipenv/)** installed


# Getting started
* Install all required packages listed in the **`pipfile`** via **`pipenv`**
* import the following packages in a pythonscript where you want to implement a Network:
    * `import pickle`
    * `import pandas as pd`
    * `from Framework_NeuralNetwork.libraries import plotting as pt`
    * `import numpy as np`
    * `from Framework_NeuralNetwork import NeuralNetwork as nn`


# Methods and functionalities of the framework
In the following section the methods of our framework are presented and explained in detail.



## `NeuralNetwork.py`: You can use this to instantiate a new NeuralNetwork object
Instantiate your network like this (after importing NeuralNetwork as nn):

    your_network = nn.neuralNetwork(activation_function_name="tanh", error_function_name="mse")

**Activation Function:** You can choose between **[TanH](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/#:~:text=The%20hyperbolic%20tangent%20activation%20function,the%20range%20%2D1%20to%201.)** (activation_function_name="tanh") and **[Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function)** (activation_function_name="sigmoid").

**Error Function:** You can choose between **[Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error)** (error_function_name="mse") if you want to solve regression tasks and **[Cross Entropy](https://en.wikipedia.org/wiki/Cross_entropy)** (error_function_name="ce") for classification tasks.



If you want to have a closer look at these functions feel free to check `logistic_regression.py` in `Framework_NeuralNetwork/libaries`.


### Fit and transform your data with our scaler
You have to scale your data with the **`scaler`** method to make sure that the network learns efficient. At first you have to fit the scaler for the data like this: 

    your_network.scaler.fit(your_data)

Then you can transform your data like this:

    your_scaled_data = your_network.scaler.transform(your_data)

Finaly, you need to create a **[one hot matrix](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)** of your ground truth like this:

    one_hot = np.identity(number_of_different_outcomes, dtype=int)[ground_truth]

### Define the architecture of your network and your thetas
Now you have to decide how the architecture of your neural net should look like. Define quantity and size of your hidden layers in an array. For example if you want to have two hidden layers with 10 neurons each you can define `your_architecture = [10, 10]`. 
If you plan to perform a simple regression task, drop the architecture variable in the method head.

Then use the **`set_thetas`** method to define the thetas of your Network and put in the architecture. Make sure to set the the number of output neurons equal to the possible outcome of your ground truth. For example like this:

    your_network.set_thetas(X=your_scaled_data, architecture=your_architecture, output_neurons=your_output_neurons)

### Finaly train your network
Now you can finaly train your network. Use the **`gradient_descent`** method. For example just like this:

    f1scores_train, error_history, accuracy, a_S = your_network.gradient_descent(X=your_scaled_data, Y=one_hot, alpha=your_learning_rate, lamb=your_lambda, iterations=number_of_iterations)

Besides your scaled data and ground truth in a one hot vector you have to set **`your_learning_rate`**. Also you have to set **`your_lambda`** as a factor for the **[regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics))** term. Eventually you have to set the **`number_of_iterations`** your network should perform.

The **`gradient_descent`** returns:
* The F1 scores of the trained data, here **`f1scores_train`**
* The error history of the trained data, here **`error_history`**
* The accuracy of the trained data, here **`accuracy`**
* and the softmax layer (last layer) of the trained network, here **`a_S`**


### Visualize your training results
Use the function **`feature_error_comp`** to visualize your error history:
    
    pt.feature_error_comp(error_histories=error_history, architecture=your_architecture)

You have to create an array with all the classes of your ground truth to set the labels of the x_axis, for example: **`your_x_axis_labels=[1,2,3]`** Then use the function **`bar_chart`** to visualize your F1 scores.

    pt.bar_chart(f1_scores=f1scores_train, x_axis_labels=your_x_axis_labels, architecture=your_architecture)

Finaly, create a confusionmatrix like 

    your_confusion_matrix=nn.confusion_matrix(a_S, one_hot)
    
and plot it:

    pt.plot_confusion(your_confusion_matrix, keys="digits", architecture=your_architecture)


### Validate your network
To make sure your network works properly you have to use it with unknown validation data. At first you have to use the scaler again to transform your validation data:

    your_validation_data_scaled = your_network.scaler.transform(your_validation_data)

Again you have to create a one hot matrix. This time with the validation data:

    validation_one_hot = np.identity(len(your_x_axis_labels), dtype=int)[ground_truth_validation]

Now you can test your networks prediction skills with the **`predict`** method:

    your_prediction=your_network.predict(your_validation_data_scaled)

### Visualize your validation results

To show the F1 score you have to create an empty array like **`f1scores_val=[]`** at first. You have to fill the array with the F1 score of each class of your ground truth. You can use a for loop with the length of **`your_axis_labels`**, calculate each F1 score with the **`f1_score`** method and append it to the **`f1scores_val`** array. Each loop could look like this:

    your_f1_score = nn.f1score(h=your_prediction, y=validation_one_hot, class_integer=i)
    f1scores_val.append(f1_score)

The class_integer wit the index (i) is used to select a certain class of the ground truth for which the F1 score is calculated.

You need to convert the F1 score array into a data frame to make it usable for the plotting function:

    f1scoress_val = pd.DataFrame(f1scores_val)

Now you can finaly plot the F1 scores of your validation data:

    pt.bar_chart(f1_scores=f1scores_val, x_axis_labels=your_x_axis_labels, architecture=your_architecture)



### If you're happy with it - save your network
Now you can use **`pickle`** to save your network:

    pickle.dump(your_network, file=open("name_of_your_network", "wb"))

If you want to use it later you can simply reload it with:

    your_network_loaded: nn.neuralNetwork = pickle.load(open("name_of_your_network", "rb"))



# Example for the usage of the neural network framework with the MNIST dataset
* In **`Example_MNIST.py`** you will find all the methods of our framework implemented with the **[MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database)**
* The script is ready to execute
* When executed it will 
    * load the MNIST data 
    * instantiate a network 
    * scale the training data 
    * train the network with the scaled data
    * show errorhistory, F1 scores and confusionmatrix of the trained data
    * scale validation data
    * use the trained network to predict digits
    * visualize F1 scores and confusionmatrix of validation data
    * saves the network
    * and finally loads it again to make it ready for further usage
 * Feel free to check out the comments in **`Example_MNIST.py`** to get into the details of the whole process
