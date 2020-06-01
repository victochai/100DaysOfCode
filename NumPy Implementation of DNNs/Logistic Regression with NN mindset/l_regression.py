#%% Libralies

import numpy as np
import os
import matplotlib.pyplot as plt
import h5py    

#%% Dataset

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

# Visualization
fig, axs = plt.subplots(5,5)
k = 0
for _ in range(5):
    for __ in range(5):
        axs[_,__].imshow(X_train_orig[k])
        axs[_,__].set_axis_off()
        if Y_train[0, k] == 0:
            axs[_,__].set_title("non-cat")
        else:
            axs[_,__].set_title("cat")
        k+=1
plt.show()
del k, axs

#%% Flatten + normalize

#X_train = X_train_orig.reshape(X_train_orig.shape[0], -1).T
#X_test = X_test_orig.reshape(X_test_orig.shape[0], -1).T
#X_train = X_train / 255
#X_test = X_test / 255
#
#print("X train shape: " + str(X_train.shape))
#print("Y train shape: " + str(Y_train.shape))
#print("X test shape: " + str(X_test.shape))
#print("Y test shape: " + str(Y_test.shape))

#%% Logistic Regression

class LogisticRegressionNN():
    """ 
    NumPy implementation of Single Layer&Single Binary Neuron Neural Network.
    Notation:
    n_x -> Number of features
    m -> Number of training examples
    m_test -> Number of testing examples
    Requirements:
    X_train.shape = (n_x, m)
    X_test.shape = (n_x, m_test)
    Y_train.shape = (1, m)
    Y_test.shape = (1, m_test)
    
    """  
    def __init__(self, num_iterations=1000, learning_rate=0.001):
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.W = None
        self.b = None
        self.m = None
        self.n_x = None

    def preprocess_rgb(self, X):
        """
        Requirements:
        X.shape = (m/m_test, width, height, 3)
        """
        return X.reshape(X.shape[0], -1).T/255
    
    def initialize_with_zeros(self, X):
            self.n_x = X.shape[0]
            self.m = X.shape[1]
            self.W = np.zeros((self.n_x, 1)) 
            self.b = 0
    
    def fit(self, X, Y):
        """ 
        Complete forward pass + cost + average partial derivates + 
        + update weights + iterate
        """ 
        costs = []
        print("Choosen learning rate : " + str(self.learning_rate))
        print("Choosen number of iterations: " + str(self.num_iterations))
        for iteration in range(0, self.num_iterations):
            # 1.) Complete forward path
            # Z = (1, n_x)*(n_x, m) + int 
            Z = np.dot(self.W.T, X) + self.b # Shape -> (1, m)
            A = 1 / (1 + np.exp(-Z)) # Shape -> (1, m) | SIGMOID      
            #2.) Average cost function  
            cost = (-1/self.m) * (np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)))
            costs.append(cost)
            if iteration % 100 == 0: 
                print("Iteration " + str(iteration) + ": " + str(cost))           
            # 3.) Global partil gradients
            # dW = (n_x, m)*(m, 1)/m = (n_x, 1)
            dW = (1/self.m) * (np.dot(X, (A-Y).T))
            db = (1/self.m) * np.sum(A-Y)          
            # 3.) Update weights
            self.W = self.W - self.learning_rate * dW
            self.b = self.b - self.learning_rate * db
        return costs
    
    def predict(self, X):
        Z = np.dot(self.W.T, X_test) + self.b 
        A = 1 / (1 + np.exp(-Z))
        for _ in range(X.shape[1]):
            if A[0, _] < 0.5:
                A[0, _] = 0
            else:
                A[0, _] = 1
        return A
    
    def accuracy(self, Y_pred, Y):
        acc = np.mean(Y_pred == Y) * 100
        print("Accuracy: " + str(acc) + " %")
        return acc

#%% Check

model = LogisticRegressionNN(num_iterations=5000)
X_train = model.preprocess_rgb(X_train_orig)
X_test = model.preprocess_rgb(X_test_orig)
model.initialize_with_zeros(X_train)
costs = model.fit(X_train, Y_train)
Y_pred = model.predict(Y_test)
acc = model.accuracy(Y_pred, Y_test)

plt.plot(costs)
plt.title("Cost function")
plt.xlabel("№ of iteration")
plt.show()

#%% Sk Learn

from sklearn import datasets
from sklearn.model_selection import train_test_split
X, Y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

Y_train = Y_train.reshape(Y_train.shape[0], 1)
Y_train = Y_train.T
Y_test = Y_test.reshape(Y_test.shape[0], 1)
Y_test = Y_test.T

X_train = X_train.T
X_test = X_test.T

model = LogisticRegressionNN(num_iterations=5000, learning_rate = 0.0001)
model.initialize_with_zeros(X_train)
costs = model.fit(X_train, Y_train)
Y_pred = model.predict(Y_test)
acc = model.accuracy(Y_pred, Y_test)

plt.plot(costs)
plt.title("Cost function")
plt.xlabel("№ of iteration")
