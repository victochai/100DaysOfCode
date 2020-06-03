#%% Libraries

import numpy as np
import os
import matplotlib.pyplot as plt
import h5py    

#%% Dataset (cat / noncat)

dr = r"D:\100DaysOfCode\NumPy Implementation of DNNs\Datasets"
tr = r"train_catvnoncat.h5"
te = r"test_catvnoncat.h5"
os.chdir(dr)

# Training data
with h5py.File(tr, "r") as f:
    train_keys = list(f.keys())    
    X_train_orig = np.array(f[train_keys[1]])
    Y_train = np.array(f[train_keys[2]]).reshape(X_train_orig.shape[0], 1).T
del train_keys

# Testing data
with h5py.File(te, "r") as f:
    test_keys = list(f.keys())    
    X_test_orig = np.array(f[test_keys[1]])
    Y_test = np.array(f[test_keys[2]]).reshape(X_test_orig.shape[0], 1).T
del test_keys

# 0 -> non-cat
# 1 -> cat
del tr, te, dr

m = X_train_orig.shape[0]

#%%

# X.shape = (n_x, m)
# Y.shape = (1, m)
# z = (5,12288)*(12288, 209) + (5,1)
layer_dims = [12288,5,3,1]
activation="relu"

def preprocess_rgb(X):
    return X.reshape(X.shape[0], -1).T/255

def initialize_with_random(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l],1))
    return parameters

def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    linear_cache = (A_prev, W, b) 
    return Z, linear_cache

def relu(Z):
    A = np.maximum(0,Z)
    activation_cache = Z
    return A, activation_cache

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    activation_cache = Z
    return A, activation_cache

def linear_activation_forward(A_prev, W, b, activation):
   if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
   elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
   cache = (linear_cache, activation_cache)
   return A, cache

def L_model_forward(X, parameters):
    caches = []
    A_prev = X
    L = len(parameters) // 2
    for l in range(1, L):
        A, cache = linear_activation_forward(
                A_prev,
                parameters["W"+str(l)],
                parameters["b"+str(l)],
                activation="relu")
        caches.append(cache)
        A_prev = A
    AL, cache = linear_activation_forward(
                A_prev,
                parameters["W"+str(L)],
                parameters["b"+str(L)],
                activation="sigmoid")
    caches.append(cache)
    return AL, caches

def cost(AL, Y):
    return np.squeeze((-1/m) * (np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))))

def linear_backward(dZ, linear_cache):
    A_prev, W, b = linear_cache
    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T, dZ)
    return dW, db, dA_prev

def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)    
    return dZ

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    dW, db, dA_prev = linear_backward(dZ, linear_cache) 
    return dW, db, dA_prev

def L_model_backward(AL, Y, caches):
    gradients = {}
    L = len(caches) 
    m = AL.shape[1]
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L-1]
    gradients["dW" + str(L)], gradients["db" + str(L)],gradients["dA" + str(L-1)] = linear_activation_backward(dAL, current_cache, activation="sigmoid")
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dW_temp, db_temp, dA_prev_temp, = linear_activation_backward(gradients["dA" + str(l+1)], current_cache, activation="relu")
        gradients["dA" + str(l)] = dA_prev_temp
        gradients["dW" + str(l + 1)] = dW_temp
        gradients["db" + str(l + 1)] = db_temp
    return gradients

def update(parameters, gradients, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        W = parameters["W" + str(l+1)]
        b = parameters["b" + str(l+1)]
        dW = gradients["dW" + str(l+1)]
        db = gradients["db" + str(l+1)]
        W = W - learning_rate * dW
        b = b - learning_rate * db
        parameters["W" + str(l+1)] = W
        parameters["b" + str(l+1)] = b
    return parameters  

def L_model(X, Y, layer_dims, num_iterations, learning_rate):
    parameters = initialize_with_random(layer_dims)
    costs = []
    for iteration in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost_ = cost(AL, Y)
        costs.append(cost_)
        if iteration % 100 == 0:
            print("Iteration" + str(iteration) + ": " + str(_cost))
        gradients = L_model_backward(AL, Y, caches)
        parameters = update(parameters, gradients, learning_rate)
    return costs
        
#%% Check
    
X_train = preprocess_rgb(X_train_orig)
X_test = preprocess_rgb(X_test_orig)

# One by one
parameters = initialize_with_random([12288, 5,5,5,3,1])
AL, cache = L_model_forward(X_train, parameters)
cost_ = cost(AL, Y_train)
gradients = L_model_backward(AL, Y_train, cache)
parameters = update(parameters, gradients, 0.01)

# ALL
costs = L_model(X_train, Y_train, [12288, 5,5,5,3,1], 2000, 0.01)

plt.plot(costs)
plt.xlabel("Iteration")
plt.title("Cost function")
