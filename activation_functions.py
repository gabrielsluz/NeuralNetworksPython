"""
Requirements: numpy
"""
import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def ReLU(x):
    return np.maximum(x, 0)

def ReLU_gradient(x):
    return (x > 0).astype(float) 

def softmax(x):
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp)