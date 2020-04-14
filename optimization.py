import numpy as np

#Adam optimizer implementation
def initialize_Adam(layers):
    """
    Argument:
    layers -- a list that stores the width of each hidden layer and of the input layer
    
    Returns:
    adam -- dictionary containing the adam values:
                    vWi -- weight matrix of shape (n_h[i], n_h[i-1])
                    vbi -- bias array of shape (n_h[i], 1)
                    v2Wi -- weight matrix of shape (n_h[i], n_h[i-1])
                    v2bi -- bias array of shape (n_h[i], 1) 
    """
    adam = {}

    for i in range(1, len(layers)):
        adam["vW" + str(i)] = np.zeros((layers[i], layers[i-1])) 
        adam["vb" + str(i)] = np.zeros((layers[i], 1))
        adam["v2W" + str(i)] = np.zeros((layers[i], layers[i-1])) 
        adam["v2b" + str(i)] = np.zeros((layers[i], 1))

    return adam

def update_parameters_Adam(adam, grads, parameters, beta1, beta2, learning_rate, num_hidden_layers, time):
    """
    Argument:
    adam -- dictionary containing the adam values:
                    vWi -- weight matrix of shape (n_h[i], n_h[i-1])
                    vbi -- bias array of shape (n_h[i], 1) 
                    v2Wi -- weight matrix of shape (n_h[i], n_h[i-1])
                    v2bi -- bias array of shape (n_h[i], 1)
    grads -- dictionary containing the gradients values:
                    dWi -- weight matrix of shape (n_h[i], n_h[i-1])
                    dbi -- bias array of shape (n_h[i], 1) 
    parameters -- dictionary containing the parameters:
                    Wi -- weight matrix of shape (n_h[i], n_h[i-1])
                    bi -- bias array of shape (n_h[i], 1) 
    beta1 -- float Parameter for momentum
    beta2 -- float Parameter for RMSProp
    layers -- a list that stores the width of each hidden layer and of the input layer
    time -- Indicates in which iteration it is
    
    Returns:
    adam -- dictionary containing the adam values:
                    vWi -- weight matrix of shape (n_h[i], n_h[i-1])
                    vbi -- bias array of shape (n_h[i], 1) 
                    v2Wi -- weight matrix of shape (n_h[i], n_h[i-1])
                    v2bi -- bias array of shape (n_h[i], 1)
    parameters -- dictionary containing the parameters:
                    Wi -- weight matrix of shape (n_h[i], n_h[i-1])
                    bi -- bias array of shape (n_h[i], 1)
    """
    epsilon = 1e-8

    for i in range(1, num_hidden_layers + 1):
        #Weighted average and bias correction
        adam["vW" + str(i)] = beta1 * adam["vW" + str(i)] + (1 - beta1) * grads["dW" + str(i)]
        adamvW_corrected = adam["vW" + str(i)] / (1 - beta1 **time )
        adam["vb" + str(i)] = beta1 * adam["vb" + str(i)] + (1 - beta1) * grads["db" + str(i)]
        adamvb_corrected = adam["vb" + str(i)] / (1 - beta1 **time )

        adam["v2W" + str(i)] = beta2 * adam["v2W" + str(i)] + (1 - beta2) * np.power(grads["dW" + str(i)], 2)    
        adamv2W_corrected = adam["v2W" + str(i)] / (1 - beta2 **time ) 
        adam["v2b" + str(i)] = beta2 * adam["v2b" + str(i)] + (1 - beta2) * np.power(grads["db" + str(i)], 2)
        adamv2b_corrected = adam["v2b" + str(i)] / (1 - beta2 **time )

        #Update parameters
        parameters["W" + str(i)] -= learning_rate * adamvW_corrected / (np.sqrt(adamv2W_corrected) + epsilon)
        parameters["b" + str(i)] -= learning_rate * adamvb_corrected / (np.sqrt(adamv2b_corrected) + epsilon)

    return adam, parameters


#Gradient descent with momentum
def initialize_Momentum(layers):
    """
    Argument:
    layers -- a list that stores the width of each hidden layer and of the input layer
    
    Returns:
    velocity -- dictionary containing the velocity values:
                    vWi -- weight matrix of shape (n_h[i], n_h[i-1])
                    vbi -- bias array of shape (n_h[i], 1)
    """
    velocity = {}

    for i in range(1, len(layers)):
        velocity["vW" + str(i)] = np.zeros((layers[i], layers[i-1])) 
        velocity["vb" + str(i)] = np.zeros((layers[i], 1))

    return velocity

def update_parameters_Momentum(velocity, grads, parameters, beta1, learning_rate, num_hidden_layers, time):
    """
    Argument:
    velocity -- dictionary containing the velocity values:
                    vWi -- weight matrix of shape (n_h[i], n_h[i-1])
                    vbi -- bias array of shape (n_h[i], 1)
    grads -- dictionary containing the gradients values:
                    dWi -- weight matrix of shape (n_h[i], n_h[i-1])
                    dbi -- bias array of shape (n_h[i], 1) 
    parameters -- dictionary containing the parameters:
                    Wi -- weight matrix of shape (n_h[i], n_h[i-1])
                    bi -- bias array of shape (n_h[i], 1) 
    beta1 -- float Parameter for momentum
    layers -- a list that stores the width of each hidden layer and of the input layer
    time -- Indicates in which iteration it is
    
    Returns:
    velocity -- dictionary containing the velocity values:
                    vWi -- weight matrix of shape (n_h[i], n_h[i-1])
                    vbi -- bias array of shape (n_h[i], 1)
    parameters -- dictionary containing the parameters:
                    Wi -- weight matrix of shape (n_h[i], n_h[i-1])
                    bi -- bias array of shape (n_h[i], 1)
    """

    for i in range(1, num_hidden_layers + 1):
        #Weighted average and bias correction
        velocity["vW" + str(i)] = beta1 * velocity["vW" + str(i)] + (1 - beta1) * grads["dW" + str(i)]
        vW_corrected = velocity["vW" + str(i)] / (1 - beta1 **time )
        velocity["vb" + str(i)] = beta1 * velocity["vb" + str(i)] + (1 - beta1) * grads["db" + str(i)]
        vb_corrected = velocity["vb" + str(i)] / (1 - beta1 **time )

        #Update parameters
        parameters["W" + str(i)] -= learning_rate * vW_corrected 
        parameters["b" + str(i)] -= learning_rate * vb_corrected 

    return velocity, parameters


#RMSprop
def initialize_RMSprop(layers):
    """
    Argument:
    layers -- a list that stores the width of each hidden layer and of the input layer
    
    Returns:
    rms -- dictionary containing the rms values:
                    v2Wi -- weight matrix of shape (n_h[i], n_h[i-1])
                    v2bi -- bias array of shape (n_h[i], 1)
    """
    rms = {}

    for i in range(1, len(layers)):
        rms["v2W" + str(i)] = np.zeros((layers[i], layers[i-1])) 
        rms["v2b" + str(i)] = np.zeros((layers[i], 1))

    return rms

def update_parameters_RMSprop(rms, grads, parameters, beta2, learning_rate, num_hidden_layers, time):
    """
    Argument:
    rms -- dictionary containing the rms values:
                    v2Wi -- weight matrix of shape (n_h[i], n_h[i-1])
                    v2bi -- bias array of shape (n_h[i], 1)
    grads -- dictionary containing the gradients values:
                    dWi -- weight matrix of shape (n_h[i], n_h[i-1])
                    dbi -- bias array of shape (n_h[i], 1) 
    parameters -- dictionary containing the parameters:
                    Wi -- weight matrix of shape (n_h[i], n_h[i-1])
                    bi -- bias array of shape (n_h[i], 1) 
    beta1 -- float Parameter for momentum
    layers -- a list that stores the width of each hidden layer and of the input layer
    time -- Indicates in which iteration it is
    
    Returns:
    rms -- dictionary containing the rms values:
                    v2Wi -- weight matrix of shape (n_h[i], n_h[i-1])
                    v2bi -- bias array of shape (n_h[i], 1)
    parameters -- dictionary containing the parameters:
                    Wi -- weight matrix of shape (n_h[i], n_h[i-1])
                    bi -- bias array of shape (n_h[i], 1)
    """
    epsilon = 1e-8

    for i in range(1, num_hidden_layers + 1):
        #Weighted average and bias correction
        rms["v2W" + str(i)] = beta2 * rms["v2W" + str(i)] + (1 - beta2) * np.power(grads["dW" + str(i)], 2)
        v2W_corrected = rms["v2W" + str(i)] / (1 - beta2 **time )
        rms["v2b" + str(i)] = beta2 * rms["v2b" + str(i)] + (1 - beta2) * np.power(grads["db" + str(i)],2)
        v2b_corrected = rms["v2b" + str(i)] / (1 - beta2 **time )

        #Update parameters
        parameters["W" + str(i)] -= learning_rate * grads["dW" + str(i)] / (np.sqrt(v2W_corrected) - epsilon)
        parameters["b" + str(i)] -= learning_rate * grads["db" + str(i)] / (np.sqrt(v2b_corrected) - epsilon)

    return rms, parameters