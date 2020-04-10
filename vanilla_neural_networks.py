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

def forwardprop_NN(X, parameters, num_hidden_layers):
    """
    Argument:
    X -- matrix of shape (n_x, m) of inputs, each column is a training example
    parameters -- dictionary containing the parameters:
                    Wi -- weight matrix of shape (n_h[i], n_h[i-1])
                    bi -- bias array of shape (n_h[i], 1)  
    num_hidden_layers -- integer
    
    Returns:
    AL -- an array of shape (1, m) containing the outputs for each example #For binary classification
     
    """
    cache = {}
    A = X

    cache["A0"] = X

    for i in range(1, num_hidden_layers):
        Z = np.dot(parameters["W" + str(i)], A) + parameters["b" + str(i)]
        A = ReLU(Z)
        cache["A" + str(i)] = A

    #Output layer
    Z = np.dot(parameters["W" + str(num_hidden_layers)], A) + parameters["b" + str(num_hidden_layers)]
    AL = sigmoid(Z)
    cache["A" + str(num_hidden_layers)] = AL

    return AL, cache



X = np.array([[1 , 2, 3], [0, 1, 2]])
Y = np.array([[0, 1, 0]])
print(X,Y)
layers = [2, 1]
parameters = initialize_parameters_rand_NN(layers)
print(parameters)
AL, cache = forwardprop_NN(X, parameters, 1)
print(AL)