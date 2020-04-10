"""
Requirements: numpy
"""
import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def ReLU(x):
    return np.maximum(x, 0)

def ReLU_gradient(x):
    return 0 if x <= 0 else 1 # Wrong