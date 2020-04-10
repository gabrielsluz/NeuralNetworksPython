"""
This file contains the implementation of a N layer Neural Network for binary classification.
The hidden units are all of the form: ReLU(np.dot(W, A) + b), execept for the last, which is sigmoid(np.dot(W, A) + b)
It will support in the future some regularization techniques and inicialization.

Requirements:   numpy
                activation_functions (file in folder)
"""
import numpy as np
from activation_functions import *

def initialize_parameters_rand_NN(layers):
    """
    Argument:
    layers -- a list that stores the width of each hidden layer and of the input layer
    
    Returns:
    parameters -- dictionary containing the parameters:
                    Wi -- weight matrix of shape (n_h[i], n_h[i-1])
                    bi -- bias array of shape (n_h[i], 1) 
    """
    parameters = {}

    for i in range(1, len(layers)):
        parameters["W" + str(i)] = np.random.randn(layers[i], layers[i-1]) * 0.01
        parameters["b" + str(i)] = np.zeros((layers[i], 1))

    return parameters



layers = [5, 4, 4 ,4, 2]
parameters = initialize_parameters_rand_NN(layers)
print(parameters)