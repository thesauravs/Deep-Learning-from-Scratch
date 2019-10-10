import numpy as np

def normaliseRows(matrix):
    matrix_norm = np.linalg.norm(matrix, axis = 1, keepdims = True)
    matrix = matrix / matrix_norm
    return matrix