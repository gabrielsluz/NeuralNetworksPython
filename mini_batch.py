import numpy as np
import math

def random_mini_batches(X, Y, mini_batch_size):
    """
    Argument:
    X -- matrix of shape (n_x, m) of inputs
    Y -- array of shape (1, m) of the correct outputs
    mini_batch_size -- integer containing the number of examples in each mini batch
    
    Returns:
    mini_batches -- a list of tuples containing pair o Xj and Yj
    """

    m = X.shape[1]
    num_batches = math.ceil(m / mini_batch_size)
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    for j in range(num_batches):
            if j*mini_batch_size >= m:
                break

            first = j*mini_batch_size
            last = min(m, (j+1)*mini_batch_size) 
            Xj = shuffled_X[:, first : last]
            Yj = shuffled_Y[:, first : last]
            mini_batch_j = (Xj, Yj)
            mini_batches.append(mini_batch_j)
    return mini_batches