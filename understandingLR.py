# Understanding Logistic Regression
# It is used for binary classification
# Lets take a mini example of binary classification
# where x1 and x2 are features and y is the output(class)
# We shall dig into the working of LR

import numpy as np

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))
                
#             Weight | class (1 = apple, 0 = non-apple)
data = np.array([[25, 1], 
                 [30, 1],
                 [100, 0],
                 [150, 0]])

# Extracting feature from data
x = data[: , 0].reshape(1, 4)
print(x)
print(x.shape)
y = data[:, 1].reshape(1, 4)
print(y)
print(y.shape)
m = 4 # number of examples

#Initializing Parameters
w = np.zeros((x.shape[0], 1))
b = 0 
print(w)
print(b)
print(w.shape)

# forward propagation
# z = W.T x + b
z = np.dot(w.T, x) + b
print(z)
# using sigmoid
a = sigmoid(z)
print(a)

# Understanding calculation of cost
print(y)
print(a)
print(np.log(a))
print(y * np.log(a))

# Calculating the cost
cost = - np.sum(( y*np.log(a) + ((1-y)*np.log(1-a)) )) / m
print(cost)

# Backward propagation
# dL/da = -(y/a) + ((1-y)/(1-a))
# dL/dz = a - y
# dL/dw = x.dz
# dL/db = dz
dz = (a - y)
dw = np.dot(x, dz.T) / m
db = np.sum(dz) / m
print(dz)
print(dw)
print(db)

# Optimize / update Parameters
learning_rate = 0.001
w = w - learning_rate * dw
b = b - learning_rate * db

print(w)
print(b)
print(w.shape)
print(b.shape)
# Predict
a = sigmoid(np.dot(w.T, x) + b)
print(a)

#---------------------------------------------------------------------
# Iteration 2
print(w)
print(x)
z = np.dot(w.T, x) + b
print(z)
# using sigmoid
a = sigmoid(z)
print(a)

cost = - np.sum(( y*np.log(a) + ((1-y)*np.log(1-a)) )) / m
print(cost)

dz = (a - y)
dw = np.dot(x, dz.T) / m
db = np.sum(dz) / m
print(dz)
print(dw)
print(db)

w = w - learning_rate * dw
b = b - learning_rate * db

print(w)
print(b)

# Predict
a = sigmoid(np.dot(w.T, x) + b)
print(a)

#---------------------------------------------------------------------
# Iteration 3
print(w)
print(x)
z = np.dot(w.T, x) + b
print(z)
# using sigmoid
a = sigmoid(z)
print(a)

cost = - np.sum(( y*np.log(a) + ((1-y)*np.log(1-a)) )) / m
print(cost)

dz = (a - y)
dw = np.dot(x, dz.T) / m
db = np.sum(dz) / m
print(dz)
print(dw)
print(db)

w = w - learning_rate * dw
b = b - learning_rate * db

print(w)
print(b)

# Predict
a = sigmoid(np.dot(w.T, x) + b)
print(a)

#--------------------------------------------------------------------
# More iterations
for i in range(50000):
    z = np.dot(w.T, x) + b
    a = sigmoid(z)
    
    cost = - np.sum(( y*np.log(a) + ((1-y)*np.log(1-a)) )) / m
    if i%500 == 0:
        print(i, cost)
    
    dz = (a - y)
    dw = np.dot(x, dz.T) / m
    db = np.sum(dz) / m
    
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    a = sigmoid(np.dot(w.T, x) + b)

print(a) # final output

#----------------------------------------------------------------------
# Testing an unkown data
test = np.array([[90],
                 [40],
                 [150],
                 [30],
                 [20]])
m_test = 5
print(test.shape)
test = test.reshape(test.shape[1], m_test)
print(w)
print(w.shape)
print(b)
result = sigmoid(np.dot(w.T, test) + b)
print(result)

# Try different learning_rate and different number of iterations to see the changes