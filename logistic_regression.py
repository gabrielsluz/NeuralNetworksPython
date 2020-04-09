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

def compute_cost_LogReg(A, Y):
    """
    Argument:
    A -- array of shape (1, m) of network outputs
    Y -- array of shape (1, m) of the correct outputs
    
    Returns:
    cost -- float
     
    """
    m = Y.shape[1]
    print(m)

    loss_array = -Y * np.log(A) - (1 - Y) * np.log(1 - A) #Loss function
    print(loss_array)
    cost = np.sum(loss_array)/m

    return cost

def backprop_LogReg(X, Y, A):
    """
    Argument:
    X -- array of shape (1, m) of inputs
    A -- array of shape (1, m) of network outputs
    Y -- array of shape (1, m) of the correct outputs
    
    Returns:
    grads -- a dictionary containing the gradient array for W and the gradiente for b
    
    """
    m = X.shape[1]

    dW = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m

    grads = {
        "dW": dW,
        "db": db
    }
    return grads



"""
W = np.reshape(np.array([2., 3., 4.]), (1, 3))
b = 1.5
params = { "W": W, "b": b}
X = np.array([[0.,1.], [0.,1.], [0., 1.]])
Y = np.reshape(np.array([1, 0]), (1, 2))
A = forwardprop_LogReg(X,params)
print("Forward step = " + str(A))
cost = compute_cost_LogReg(A, Y)
print("cost = " + str(cost))
"""
