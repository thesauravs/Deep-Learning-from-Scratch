import numpy as np

def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    sftmx = x_exp / x_sum
    return sftmx