"""sigmoid (x) = 1/(1+e^(-x)) 
   sigmoig_derivative (x) = x(1-x) """
import math
import numpy as np

def sigmoid_math(x):
    s = 1 / (1 + math.exp(-x))
    return s

def sigmoid_np(x):
    s = 1 / (1 + np.exp(-x))
    return s