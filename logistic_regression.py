"""
Requirements:   numpy
                activation_functions (file in folder)
"""
import numpy as np
from activation_functions import sigmoid

def initialize_parameters_rand_LogReg(n_x):
    """
    Argument:
    n_x -- size of the input layer
    
    Returns:
    parameters -- dictionary containing the parameters:
                    W -- weight vector of shape (1, n_x)
                    b -- bias float 
    """
    W = np.random.randn(1, n_x) * 0.01
    b = 0.0

    parameters = {
        "W": W,
        "b": b
    }

    return parameters


def forwardprop_LogReg(X, parameters):
    """
    Argument:
    X -- matrix of shape (n_x, m) of inputs, each column is a training example
    parameters -- dictionary containing the parameters:
                    W -- weight vector of shape (1, n_x)
                    b -- bias float 
    
    Returns:
    A -- an array of shape (1, m) containing the outputs for each example 
     
    """
    W = parameters['W']
    b = parameters['b']

    Z = np.dot(W,X) + b
    A = sigmoid(Z)

    return A
"""
W = np.reshape(np.array([2., 3., 4.]), (1, 3))
b = 1.5
params = { "W": W, "b": b}
X = np.array([[0.,1.], [0.,1.], [0., 1.]])
print(forwardprop_LogReg(X,params))
print(sigmoid(1.5))
print(sigmoid(10.5))
"""
