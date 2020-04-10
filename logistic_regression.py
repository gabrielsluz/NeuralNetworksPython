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
                    W -- weight vector of shape (n_x, 1)
                    b -- bias float 
    """
    W = np.random.randn(n_x, 1) * 0.01
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
                    W -- weight vector of shape (n_x, 1)
                    b -- bias float 
    
    Returns:
    A -- an array of shape (1, m) containing the outputs for each example 
     
    """
    W = parameters['W']
    b = parameters['b']

    Z = np.dot(W.T, X) + b
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
    epsilon = 0.000001
    m = Y.shape[1]

    loss_array = -Y * np.log(A + epsilon) - (1 - Y) * np.log(1 - A + epsilon) #Loss function with epsilon to avoid log(0)
    cost = np.sum(loss_array)/m

    return cost

def backprop_LogReg(X, Y, A):
    """
    Argument:
    X -- matrix of shape (n_x, m) of inputs
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

def update_parameters_LogReg(parameters, grads, learning_rate):
    """
    Argument:
    parameters -- dictionary containing the parameters:
                    W -- weight vector of shape (n_x, 1)
                    b -- bias float 
    grads -- a dictionary containing the gradient array for W and the gradiente for b
    learning_rate -- float 
    
    Returns:
    parameters -- dictionary containing the parameters:
                    W -- weight vector of shape (n_x, 1)
                    b -- bias float 
    
    """

    parameters["W"] -= learning_rate * grads["dW"]
    parameters["b"] -= learning_rate * grads["db"]

    return parameters

def LogReg(X, Y, learning_rate, num_iterations, print_cost = False, print_every = 100, load_parameters = False, loaded_parameters = {}):
    """
    Argument:
    X -- matrix of shape (n_x, m) of inputs
    Y -- array of shape (1, m) of the correct outputs
    learning_rate -- float
    num_iterations -- integer that defines how many iterations of gradient descent
    print_cost = True prints cost
    
    Returns:
    parameters -- dictionary containing the computed parameters:
                    W -- weight vector of shape (n_x, 1)
                    b -- bias float 
    costs -- array with costs from every print_every iterations
    
    """
    n_x = X.shape[0]
    if(load_parameters):
        parameters = loaded_parameters
    else:
        parameters = initialize_parameters_rand_LogReg(n_x)
    

    costs = []

    for i in range(num_iterations):
        A = forwardprop_LogReg(X, parameters)

        if(i % print_every == 0):
            costs.append(compute_cost_LogReg(A, Y))
            if(print_cost):
                print("Cost in iteration " + str(i) + " is = " + str(costs[i // print_every]))

        grads = backprop_LogReg(X, Y, A)
        parameters = update_parameters_LogReg(parameters, grads, learning_rate)
    
    return parameters, costs


def predict_LogReg(X, parameters):
    """
    Returns the predictions for the input X

    Argument:
    X -- matrix of shape (n_x, m) of inputs
    parameters -- dictionary containing the parameters:
                    W -- weight vector of shape (n_x, 1)
                    b -- bias float
    
    Returns:
    predictions -- array of shape (1, m) of the predicted outputs
    
    """
    A = forwardprop_LogReg(X, parameters)

    m = A.shape[1]

    predictions = np.zeros((1, m))

    for i in range(m):
        if A[0,i] <= 0.5:
            predictions[0,i] = 0
        else:
            predictions[0,i] = 1
    
    return predictions
    
