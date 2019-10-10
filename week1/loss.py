import numpy as np

def l1(yhat, y):
    loss = np.sum(y - yhat)
    return loss

def l2(yhat, y):
    loss = np.sum(np.dot(y - yhat, y - yhat))
    return loss