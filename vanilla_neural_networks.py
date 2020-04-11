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
    As -- a list containing the As of each layer
    Zs -- a list containing the Zs of each layer
     
    """
    As = []
    Zs = []
    A = X

    As.append(X)
    Zs.append(0.0) #Just to adjust index

    for i in range(1, num_hidden_layers):
        Z = np.dot(parameters["W" + str(i)], A) + parameters["b" + str(i)]
        A = ReLU(Z)
        As.append(A)
        Zs.append(Z)

    #Output layer
    Z = np.dot(parameters["W" + str(num_hidden_layers)], A) + parameters["b" + str(num_hidden_layers)]
    A = sigmoid(Z)
    As.append(A)
    Zs.append(Z)

    return As, Zs


def backprop_NN(As, Zs, Y, parameters, num_hidden_layers):
    """
    Argument:
    As -- a list containing the As of each layer
    Zs -- a list containing the Zs of each layer
    Y -- the labels
    parameters -- dictionary containing the parameters:
                    Wi -- weight matrix of shape (n_h[i], n_h[i-1])
                    bi -- bias array of shape (n_h[i], 1)  
    num_hidden_layers -- integer
    
    Returns:
    grads -- a dictionary with the gradients for each layer
     
    """
    #d{something} means the partial derivative of the cost function in respect to something
    m = As[0].shape[1]
    grads = {}
    #For the output sigmoid layer:
    dZ = (As[num_hidden_layers] - Y)/m #Change here if changed the output activation function

    for i in range(num_hidden_layers, 0, -1):
        grads["dW" + str(i)] = np.dot(dZ, As[i - 1].T)
        grads["db" + str(i)] = np.sum(dZ)
        #For the next layer (i-1):
        dA = np.dot(parameters["W" + str(i)].T, dZ)
        dZ = dA * ReLU_gradient(Zs[i])
    
    return grads

def update_parameters_NN(parameters, grads, learning_rate, num_hidden_layers):
    """
    Argument:
    parameters -- dictionary containing the parameters:
                    Wi -- weight matrix of shape (n_h[i], n_h[i-1])
                    bi -- bias array of shape (n_h[i], 1)b
    grads -- a dictionary with the gradients for each layer
    learning_rate -- float 
    
    Returns:
    parameters -- dictionary containing the parameters:
                    Wi -- weight matrix of shape (n_h[i], n_h[i-1])
                    bi -- bias array of shape (n_h[i], 1)b
    
    """
    for i in range(1, num_hidden_layers + 1):
        parameters["W" + str(i)] -= learning_rate * grads["dW" + str(i)]
        parameters["b" + str(i)] -= learning_rate * grads["db" + str(i)]

    return parameters


def model_NN(X, Y, layers, learning_rate, num_iterations, print_cost = False, print_every = 100, load_parameters = False, loaded_parameters = {}):
    """
    Argument:
    X -- matrix of shape (n_x, m) of inputs
    Y -- array of shape (1, m) of the correct outputs
    layers -- (n_x,n_h[1], ... ,n_h[n]) The width of each layer, including input and output
    learning_rate -- float
    num_iterations -- integer that defines how many iterations of gradient descent
    print_cost = True prints cost
    
    Returns:
    parameters -- dictionary containing the parameters:
                    Wi -- weight matrix of shape (n_h[i], n_h[i-1])
                    bi -- bias array of shape (n_h[i], 1)b
    costs -- array with costs from every print_every iterations
    
    """
    if(load_parameters):
        parameters = loaded_parameters
    else:
        parameters = initialize_parameters_rand_NN(layers)
    

    costs = []
    num_hidden_layers = len(layers) - 1

    for i in range(num_iterations):
        As, Zs = forwardprop_NN(X, parameters, num_hidden_layers)

        if(i % print_every == 0):
            costs.append(compute_cost_Sigmoid_NN(A, Y))
            if(print_cost):
                print("Cost in iteration " + str(i) + " is = " + str(costs[i // print_every]))

        grads = backprop_NN(As, Zs, Y, parameters, num_hidden_layers)
        parameters = update_parameters_LogReg(parameters, grads, learning_rate, num_hidden_layers)
    
    return parameters, costs

def predict_NN(X, parameters, num_hidden_layers):
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
    As, Zs = forwardprop_NN(X, parameters, num_hidden_layers)

    m = X.shape[1]

    predictions = np.zeros((1, m))

    for i in range(m):
        if As[num_hidden_layers][0,i] <= 0.5:
            predictions[0,i] = 0
        else:
            predictions[0,i] = 1
    
    return predictions

def compute_cost_Sigmoid_NN(A, Y):
    """
    Argument:
    A -- array of shape (1, m) of network outputs
    Y -- array of shape (1, m) of the correct outputs
    
    Returns:
    cost -- float
     
    """
    epsilon = 0.000000000001
    m = Y.shape[1]

    loss_array = -Y * np.log(A + epsilon) - (1 - Y) * np.log(1 - A + epsilon) #Loss function with epsilon to avoid log(0)
    cost = np.sum(loss_array)/m

    return cost

def gradient_checking(X, Y, num_hidden_layers, parameters):
    As, Zs = forwardprop_NN(X, parameters, num_hidden_layers)
    grads = backprop_NN(As, Zs, Y, parameters, num_hidden_layers)

    aprox_grads = {}

    epsilon = 0.00000000001

    for i in range(num_hidden_layers):
        #W
        W_array = np.zeros(parameters["W" + str(i+1)].shape)
        for j in range(parameters["W" + str(i+1)].shape[0]):
            for k in range(parameters["W" + str(i+1)].shape[1]):
                parameters["W" + str(i+1)][j][k] += epsilon
                As, Zs = forwardprop_NN(X, parameters, num_hidden_layers)
                cost_plus = compute_cost_Sigmoid_NN(As[num_hidden_layers], Y)

                parameters["W" + str(i+1)][j][k] -= 2*epsilon
                As, Zs = forwardprop_NN(X, parameters, num_hidden_layers)
                cost_minus = compute_cost_Sigmoid_NN(As[num_hidden_layers], Y)

                W_array[j][k] = (cost_plus - cost_minus) / (2*epsilon)
                parameters["W" + str(i+1)][j][k] += epsilon
        aprox_grads["dW" + str(i+1)] = W_array
        #b
        b_array = np.zeros((parameters["W" + str(i+1)].shape[0],1))
        for j in range(parameters["W" + str(i+1)].shape[0]):
            parameters["b" + str(i+1)][j] += epsilon
            As, Zs = forwardprop_NN(X, parameters, num_hidden_layers)
            cost_plus = compute_cost_Sigmoid_NN(As[num_hidden_layers], Y)

            parameters["b" + str(i+1)][j] -= 2*epsilon
            As, Zs = forwardprop_NN(X, parameters, num_hidden_layers)
            cost_minus = compute_cost_Sigmoid_NN(As[num_hidden_layers], Y)

            b_array[j] = (cost_plus - cost_minus) / (2*epsilon)
            parameters["b" + str(i+1)][j] += epsilon
        aprox_grads["db" + str(i+1)] = b_array

    print(grads)
    print(aprox_grads)




X = np.array([[1 , 2, 3], [0, 1, 2]])
Y = np.array([[0, 1, 0]])
#print(X,Y)
layers = [2, 1]
parameters = initialize_parameters_rand_NN(layers)
#print(parameters)
As, Zs = forwardprop_NN(X, parameters, 1)
#print(Zs[1], As[1])
grads = backprop_NN(As, Zs, Y, parameters, 1)
#print(grads)
gradient_checking(X, Y, 1, parameters)