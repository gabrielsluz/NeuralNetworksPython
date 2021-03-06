"""
This file contains the implementation of a N layer Neural Network for multi class classification.
The hidden units are all of the form: ReLU(np.dot(W, A) + b), execept for the last, which is softmax(np.dot(W, A) + b)

Requirements:   numpy
                activation_functions (file in folder)
                weight_initializations (file in folder)
                optimization (file in folder)
"""
import numpy as np
from activation_functions import *
from weight_initializations import *
from optimization import *
from mini_batch import *


def forwardprop_SNN(X, parameters, num_hidden_layers):
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

    #Output layer (Softmax)
    Z = np.dot(parameters["W" + str(num_hidden_layers)], A) + parameters["b" + str(num_hidden_layers)]
    A = softmax(Z)
    As.append(A)
    Zs.append(Z)

    return As, Zs


def backprop_SNN(As, Zs, Y, parameters, num_hidden_layers):
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
    #For the output softmax layer:
    dZ = (As[num_hidden_layers] - Y)/m #Change here if changed the output activation function

    for i in range(num_hidden_layers, 0, -1):
        grads["dW" + str(i)] = np.dot(dZ, As[i - 1].T)
        grads["db" + str(i)] = np.sum(dZ, axis=1, keepdims=True)

        #For the next layer (i-1):
        if i > 1:
            dA = np.dot(parameters["W" + str(i)].T, dZ)
            dZ = dA * ReLU_gradient(Zs[i-1])
    
    return grads

def update_parameters_SNN(parameters, grads, learning_rate, num_hidden_layers):
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

def predict_SNN(X, parameters, num_hidden_layers):
    """
    Returns the predictions for the input X

    Argument:
    X -- matrix of shape (n_x, m) of inputs
    parameters -- dictionary containing the parameters:
                    W -- weight vector of shape (n_x, 1)
                    b -- bias float
    
    Returns:
    predictions -- array of shape (num_classes, m) of the predicted outputs
    
    """
    As, _ = forwardprop_SNN(X, parameters, num_hidden_layers)
    output = As[num_hidden_layers]
    num_classes = output.shape[0]
    m = X.shape[1]

    predictions = np.zeros((num_classes, m))

    indexes = np.argmax(output, axis = 0)
    columns = np.arange(m)

    predictions[indexes, columns] = 1
    
    return predictions

def compute_cost_Softmax_NN(A, Y):
    """
    Argument:
    A -- matrix of shape (num_classes, m) of network outputs
    Y -- matrix of shape (num_classes, m) of the correct outputs
    
    Returns:
    cost -- float
     
    """
    epsilon = 0.000000000001
    m = Y.shape[1]

    loss_array = -Y * np.log(A + epsilon) #Loss function with epsilon to avoid log(0)
    cost = np.sum(loss_array)/m

    return cost

def gradient_checking_SNN(X, Y, num_hidden_layers, parameters):    
    """
    It does not work with dropout

    Argument:
    X -- matrix of shape (n_x, m) of inputs
    Y -- matrix of shape (num_classes, m) of the correct outputs
    num_hidden_layers -- integer
    parameters -- dictionary containing the parameters
    
    Returns:
    cost -- float
     
    """

    As, Zs = forwardprop_SNN(X, parameters, num_hidden_layers)
    grads = backprop_SNN(As, Zs, Y, parameters, num_hidden_layers)

    aprox_grads = {}

    epsilon = 0.0000001

    for i in range(num_hidden_layers):
        #W
        W_array = np.zeros(parameters["W" + str(i+1)].shape)
        for j in range(parameters["W" + str(i+1)].shape[0]):
            for k in range(parameters["W" + str(i+1)].shape[1]):
                parameters["W" + str(i+1)][j][k] += epsilon
                As, Zs = forwardprop_SNN(X, parameters, num_hidden_layers)
                cost_plus = compute_cost_Softmax_NN(As[num_hidden_layers], Y)

                parameters["W" + str(i+1)][j][k] -= 2*epsilon
                As, Zs = forwardprop_SNN(X, parameters, num_hidden_layers)
                cost_minus = compute_cost_Softmax_NN(As[num_hidden_layers], Y)

                W_array[j][k] = (cost_plus - cost_minus) / (2*epsilon)
                parameters["W" + str(i+1)][j][k] += epsilon
        aprox_grads["dW" + str(i+1)] = W_array
        #b
        b_array = np.zeros(parameters["b" + str(i+1)].shape)
        for j in range(parameters["b" + str(i+1)].shape[0]):
            parameters["b" + str(i+1)][j] += epsilon
            As, Zs = forwardprop_SNN(X, parameters, num_hidden_layers)
            cost_plus = compute_cost_Softmax_NN(As[num_hidden_layers], Y)

            parameters["b" + str(i+1)][j] -= 2*epsilon
            As, Zs = forwardprop_SNN(X, parameters, num_hidden_layers)
            cost_minus = compute_cost_Softmax_NN(As[num_hidden_layers], Y)

            b_array[j] = (cost_plus - cost_minus) / (2*epsilon)
            parameters["b" + str(i+1)][j] += epsilon
        aprox_grads["db" + str(i+1)] = b_array
    print("Computed grads :")
    print(grads)
    print("Aproximated grads:")
    print(aprox_grads)



def forwardprop_dropout_SNN(X, parameters, num_hidden_layers, keep_prob):
    """
    Argument:
    X -- matrix of shape (n_x, m) of inputs, each column is a training example
    parameters -- dictionary containing the parameters:
                    Wi -- weight matrix of shape (n_h[i], n_h[i-1])
                    bi -- bias array of shape (n_h[i], 1)  
    num_hidden_layers -- integer
    keep_prob -- Probability of keeping a neuron in dropout
    
    Returns:
    As -- a list containing the As of each layer
    Zs -- a list containing the Zs of each layer
     
    """
    As = []
    Zs = []
    Ds = []
    A = X

    As.append(X)
    Zs.append(0.0) #Just to adjust index
    Ds.append(0.0) #Just to adjust index

    for i in range(1, num_hidden_layers):
        Z = np.dot(parameters["W" + str(i)], A) + parameters["b" + str(i)]
        A = ReLU(Z)

        D = (np.random.rand(A.shape[0], A.shape[1]) < keep_prob).astype(int)
        A = (A*D) / keep_prob

        As.append(A)
        Zs.append(Z)
        Ds.append(D)

    #Output layer (Softmax)
    Z = np.dot(parameters["W" + str(num_hidden_layers)], A) + parameters["b" + str(num_hidden_layers)]
    A = softmax(Z)
    As.append(A)
    Zs.append(Z)

    return As, Zs, Ds

def backprop_dropout_SNN(As, Zs, Ds, Y, parameters, num_hidden_layers, keep_prob):
    """
    Argument:
    As -- a list containing the As of each layer
    Zs -- a list containing the Zs of each layer
    Ds -- a list containing the Ds of each layer
    Y -- the labels
    parameters -- dictionary containing the parameters:
                    Wi -- weight matrix of shape (n_h[i], n_h[i-1])
                    bi -- bias array of shape (n_h[i], 1)  
    num_hidden_layers -- integer
    keep_prob -- Probability of keeping a neuron in dropout
    
    Returns:
    grads -- a dictionary with the gradients for each layer
     
    """
    #d{something} means the partial derivative of the cost function in respect to something
    m = As[0].shape[1]
    grads = {}
    #For the output softmax layer:
    dZ = (As[num_hidden_layers] - Y)/m #Change here if changed the output activation function

    for i in range(num_hidden_layers, 0, -1):
        grads["dW" + str(i)] = np.dot(dZ, As[i - 1].T)
        grads["db" + str(i)] = np.sum(dZ, axis=1, keepdims=True)

        #For the next layer (i-1):
        if i > 1:
            dA = np.dot(parameters["W" + str(i)].T, dZ)
            dA = (dA * Ds[i-1]) / keep_prob
            dZ = dA * ReLU_gradient(Zs[i-1])
    
    return grads



#Main function
def model_SNN(X, Y, layers, learning_rate = 0.5, num_iterations = 1000, print_cost = False, print_every = 100, initialization = "rand", loaded_parameters = {}):
    """
    Argument:
    X -- matrix of shape (n_x, m) of inputs
    Y -- array of shape (1, m) of the correct outputs
    layers -- (n_x,n_h[1], ... ,n_h[n]) The width of each layer, including input and output
    learning_rate -- float
    num_iterations -- integer that defines how many iterations of gradient descent
    print_cost = True prints cost
    initialization -- String the informs the type of weight initalization :
                        rand -- Random *0.01
                        he -- He intialization
                        load -- Load the parameters with loaded_parameters
    
    Returns:
    parameters -- dictionary containing the parameters:
                    Wi -- weight matrix of shape (n_h[i], n_h[i-1])
                    bi -- bias array of shape (n_h[i], 1)
    costs -- array with costs from every print_every iterations
    
    """
    if(initialization == "load"):
        parameters = loaded_parameters
    elif(initialization == "he"):
        parameters = initialize_parameters_He_NN(layers)
    elif(initialization == "rand"):
        parameters = initialize_parameters_rand_NN(layers)
    else:
        print("Invalid initialization. Exiting")
        return
    

    costs = []
    num_hidden_layers = len(layers) - 1

    for i in range(num_iterations):
        As, Zs = forwardprop_SNN(X, parameters, num_hidden_layers)

        if(i % print_every == 0):
            costs.append(compute_cost_Softmax_NN(As[num_hidden_layers], Y))
            if(print_cost):
                print("Cost in iteration " + str(i) + " is = " + str(costs[i // print_every]))

        grads = backprop_SNN(As, Zs, Y, parameters, num_hidden_layers)
        parameters = update_parameters_SNN(parameters, grads, learning_rate, num_hidden_layers)
    
    return parameters, costs


#Main function for mini batches
def model_mini_batch_SNN(X, Y, layers, learning_rate = 0.5, mini_batch_size = 512, num_iterations = 1000, print_cost = False, print_every = 100, initialization = "rand", loaded_parameters = {}):
    """
    Argument:
    X -- matrix of shape (n_x, m) of inputs
    Y -- array of shape (1, m) of the correct outputs
    layers -- (n_x,n_h[1], ... ,n_h[n]) The width of each layer, including input and output
    mini_batch_size -- integer that indicates the size of each mini batch
    learning_rate -- float
    num_iterations -- integer that defines how many iterations of gradient descent
    print_cost = True prints cost
    initialization -- String the informs the type of weight initalization :
                        rand -- Random *0.01
                        he -- He intialization
                        load -- Load the parameters with loaded_parameters
    
    Returns:
    parameters -- dictionary containing the parameters:
                    Wi -- weight matrix of shape (n_h[i], n_h[i-1])
                    bi -- bias array of shape (n_h[i], 1)
    costs -- array with costs from every print_every iterations
    
    """
    if(initialization == "load"):
        parameters = loaded_parameters
    elif(initialization == "he"):
        parameters = initialize_parameters_He_NN(layers)
    elif(initialization == "rand"):
        parameters = initialize_parameters_rand_NN(layers)
    else:
        print("Invalid initialization. Exiting")
        return
    

    costs = []
    num_hidden_layers = len(layers) - 1

    mini_batches = random_mini_batches(X, Y, mini_batch_size)
    num_batches = len(mini_batches)

    for i in range(num_iterations):
        for j in range(num_batches):
            Xj, Yj = mini_batches[j]

            As, Zs = forwardprop_SNN(Xj, parameters, num_hidden_layers)

            grads = backprop_SNN(As, Zs, Yj, parameters, num_hidden_layers)
            parameters = update_parameters_SNN(parameters, grads, learning_rate, num_hidden_layers)
            
        if(i % print_every == 0):
                costs.append(compute_cost_Softmax_NN(As[num_hidden_layers], Yj))
                if(print_cost):
                    print("Cost in iteration " + str(i) + " is = " + str(costs[i // print_every]))
    
    return parameters, costs


#Main function for mini batches with dropout
def model_mini_batch_dropout_SNN(X, Y, layers, learning_rate = 0.5,  keep_prob = 0.8, mini_batch_size = 512, num_iterations = 1000, print_cost = False, print_every = 100, initialization = "rand", loaded_parameters = {}):
    """
    Argument:
    X -- matrix of shape (n_x, m) of inputs
    Y -- array of shape (1, m) of the correct outputs
    layers -- (n_x,n_h[1], ... ,n_h[n]) The width of each layer, including input and output
    mini_batch_size -- integer that indicates the size of each mini batch
    learning_rate -- float
    keep_prob -- Probability of keeping a neuron in dropout
    num_iterations -- integer that defines how many iterations of gradient descent
    print_cost = True prints cost
    initialization -- String the informs the type of weight initalization :
                        rand -- Random *0.01
                        he -- He intialization
                        load -- Load the parameters with loaded_parameters
    
    Returns:
    parameters -- dictionary containing the parameters:
                    Wi -- weight matrix of shape (n_h[i], n_h[i-1])
                    bi -- bias array of shape (n_h[i], 1)
    costs -- array with costs from every print_every iterations
    
    """
    if(initialization == "load"):
        parameters = loaded_parameters
    elif(initialization == "he"):
        parameters = initialize_parameters_He_NN(layers)
    elif(initialization == "rand"):
        parameters = initialize_parameters_rand_NN(layers)
    else:
        print("Invalid initialization. Exiting")
        return
    

    costs = []
    num_hidden_layers = len(layers) - 1

    mini_batches = random_mini_batches(X, Y, mini_batch_size)
    num_batches = len(mini_batches)

    for i in range(num_iterations):
        for j in range(num_batches):
            Xj, Yj = mini_batches[j]

            As, Zs, Ds = forwardprop_dropout_SNN(Xj, parameters, num_hidden_layers, keep_prob)

            grads = backprop_dropout_SNN(As, Zs, Ds, Yj, parameters, num_hidden_layers, keep_prob)
            parameters = update_parameters_SNN(parameters, grads, learning_rate, num_hidden_layers)
            
        if(i % print_every == 0):
                costs.append(compute_cost_Softmax_NN(As[num_hidden_layers], Yj))
                if(print_cost):
                    print("Cost in iteration " + str(i) + " is = " + str(costs[i // print_every]))
    
    return parameters, costs

#Main function for mini batches with dropout and different optimization algorithms
def model_dropout_Opt_SNN(X, Y, layers, learning_rate = 0.5,  keep_prob = 0.8, mini_batch_size = 512, beta1 = 0.9, beta2 = 0.999, optimizer = "momentum", num_iterations = 1000, print_cost = False, print_every = 100, initialization = "rand", loaded_parameters = {}):
    """
    Argument:
    X -- matrix of shape (n_x, m) of inputs
    Y -- array of shape (1, m) of the correct outputs
    layers -- (n_x,n_h[1], ... ,n_h[n]) The width of each layer, including input and output
    mini_batch_size -- integer that indicates the size of each mini batch
    beta1 -- float Parameter for momentum
    beta2 -- float Parameter for RMSProp
    optimizer  -- string that defines which optimizer to use:
                    adam
                    rms
                    momentum
    learning_rate -- float
    keep_prob -- Probability of keeping a neuron in dropout
    num_iterations -- integer that defines how many iterations of gradient descent
    print_cost = True prints cost
    initialization -- String the informs the type of weight initalization :
                        rand -- Random *0.01
                        he -- He intialization
                        load -- Load the parameters with loaded_parameters
    
    Returns:
    parameters -- dictionary containing the parameters:
                    Wi -- weight matrix of shape (n_h[i], n_h[i-1])
                    bi -- bias array of shape (n_h[i], 1)
    costs -- array with costs from every print_every iterations
    
    """
    if(initialization == "load"):
        parameters = loaded_parameters
    elif(initialization == "he"):
        parameters = initialize_parameters_He_NN(layers)
    elif(initialization == "rand"):
        parameters = initialize_parameters_rand_NN(layers)
    else:
        print("Invalid initialization. Exiting")
        return
    
    if(optimizer == "adam"):
        opt_params = initialize_Adam(layers)
        opt_update_params = update_parameters_Adam
    elif(optimizer == "rms"):
        opt_params = initialize_RMSprop(layers)
        opt_update_params = update_parameters_RMSprop
    elif(optimizer == "momentum"):
        opt_params = initialize_Momentum(layers)
        opt_update_params = update_parameters_Momentum
    else:
        print("Invalid Optimizer. Exiting")
        return

    time = 1 #Must be 1, if 0 then division by zero will occur in update_parameters_adam

    costs = []
    num_hidden_layers = len(layers) - 1

    mini_batches = random_mini_batches(X, Y, mini_batch_size)
    num_batches = len(mini_batches)

    for i in range(num_iterations):
        for j in range(num_batches):
            Xj, Yj = mini_batches[j]

            As, Zs, Ds = forwardprop_dropout_SNN(Xj, parameters, num_hidden_layers, keep_prob)

            grads = backprop_dropout_SNN(As, Zs, Ds, Yj, parameters, num_hidden_layers, keep_prob)

            opt_params, parameters = opt_update_params(opt_params, grads, parameters, beta1, beta2, learning_rate, num_hidden_layers, time)
            time += 1
            
        if(i % print_every == 0):
                costs.append(compute_cost_Softmax_NN(As[num_hidden_layers], Yj))
                if(print_cost):
                    print("Cost in iteration " + str(i) + " is = " + str(costs[i // print_every]))
    
    return parameters, costs

"""
X = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
Y = np.array([[0, 1, 0, 0], [1, 0, 1, 1]])
#print(X,Y)
layers = [2, 3, 2]
num_hidden_layers = len(layers) - 1

parameters, costs = model_dropout_Adam_SNN(X, Y, layers, learning_rate = 0.5,  keep_prob = 0.6, mini_batch_size = 512, beta1 = 0.9, beta2 = 0.999, num_iterations = 50, print_cost = True, print_every = 5, initialization = "rand", loaded_parameters = {})


parameters = initialize_parameters_rand_NN(layers)

adam = initialize_Adam(layers)
time = 1
#print(adam)
As, Zs, Ds = forwardprop_dropout_SNN(X, parameters, num_hidden_layers, 0.8)
grads = backprop_dropout_SNN(As, Zs, Ds, Y, parameters, num_hidden_layers, 0.8)
print(grads)

adam, parameters = update_parameters_Adam(adam, grads, parameters, 0.9, 0.99, 0.5, num_hidden_layers, time)
print(adam)
#

parameters = initialize_parameters_rand_NN(layers)
print(parameters)


#As, Zs = forwardprop_SNN(X, parameters, num_hidden_layers)
As, Zs, Ds = forwardprop_dropout_SNN(X, parameters, num_hidden_layers, 0.8)
print(Ds)

grads = backprop_dropout_SNN(As, Zs, Ds, Y, parameters, num_hidden_layers, 0.8)
#print(grads)

parameters = update_parameters_SNN(parameters, grads, 0.5, num_hidden_layers)

gradient_checking_SNN(X, Y, num_hidden_layers, parameters)

X = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
Y = np.array([[0, 1, 0, 0, 1], [1, 0, 1, 1, 0]])
print(random_mini_batches(X, Y, 2))
"""
