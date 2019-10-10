import numpy as np
import matplotlib.pyplot as plt
#import h5py
#import scipy
#from PIL import Image
#from scipy import ndimage
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
plt.imshow(train_set_x_orig[22])

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

#To check dimension of train and test images
print(train_set_x_orig.shape)
print(train_set_y.shape)

print(test_set_x_orig.shape)
print(test_set_y.shape)

#To flatten the images into (num_px * num_px * 3, 1)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

#Standardize the data 
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

# 1. Define model structure
# 2. Initialise model's parameters
# 3. Loop : forward propagation, backward propagation, updating parameters
def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def initialize_with_zeros(dim):
    #Arguments: dimensions
    w = np.zeros((dim, 1))
    b = 0
    
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

# Forward propagation (computing cost)
    # A = σ(w.T * X + b) = (a(1), a(2), ..., a(m−1), a(m))
    # J = −1/m ∑(i = 1 to m) y(i) * log(a(i)) + (1 − y(i)) log(1 − a(i))

# Backward propagation (computing gradient)
    # ∂J / ∂w = 1/m X * (A−Y).T
    # ∂J / ∂b = 1/m ∑(i = 1 to m) (a(i)−y(i))
    
def propagate(w, b, X, Y):
    """
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
    """
    m = X.shape[1] #number of examples
    
    #Forward
    A = sigmoid(np.dot(w.T, X) + b)
    cost = np.sum(- (Y * np.log(A)) - ((1 - Y) * np.log(1 - A))) / m
    
    #Backward
    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

# Update the parameters
def optimize(w, b, X, Y, num_iter, learning_rate, print_cost = False):
    costs = []
    
    for i in range(num_iter):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)
        
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))
        
    params = {"w" : w,
              "b" : b}
    grads = {"dw" : dw,
             "db" :db}
    
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_pred = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        if A[0][i] <= 0.5:
            Y_pred[0][i] = 0
        else:
            Y_pred[0][i] = 1
            
    assert(Y_pred.shape == (1, m))
    return Y_pred

def model(X_train, Y_train, X_test, Y_test, num_iter = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    w, b = initialize_with_zeros(X_train.shape[0])
    
    parameters, grads, cost = optimize(w, b, X_train, Y_train, num_iter, learning_rate, print_cost = True)
    
    w = parameters["w"]
    b = parameters["b"]
    
    Y_predict_test = predict(w, b, X_test)
    Y_predict_train = predict(w, b, X_train)
    
    print("Train accuracy")
    print (100 - np.mean(np.abs(Y_predict_train - Y_train)) * 100)
    print("Test accuracy")
    print (100 - np.mean(np.abs(Y_predict_test - Y_test)) * 100)
    
    d = {"costs" : cost,
         "Y_predict_train": Y_predict_train,
         "Y_predict_test": Y_predict_test,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iter" : num_iter
         }
    
    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iter = 10000, learning_rate = 0.0001, print_cost = True)

#print(test_set_y)
#print(d["Y_predict_test"])
index = 12
plt.imshow(test_set_x[:,index].reshape(num_px, num_px, 3))
print("y = " + str(test_set_y[0, index]) + " and predicted = " + str(d["Y_predict_test"][0,index]))

#plot the cost vs num_iteration
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()