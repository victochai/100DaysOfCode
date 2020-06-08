#%% Libraries

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

#%% Perceptron

class Perceptron():
    """ Perceptron implementation. """
    
    def __init__(self, lr=0.01, num_iterations=50):
        """
        Parameters:
        lr - learning rate
        num_iteration - number of iterations
        """
        self.lr = lr
        self.num_iterations = num_iterations
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        """ 
        Notation:
        n_x - num. of features
        m - num. of training examples
        Parameters:
        X train of shape (m, n_x)
        y train of shape (m,) or (m, 1)
        """
        m, n_x = X.shape
        # 1.) Initialize weights
        self.w = np.zeros(n_x)
        self.b = 0.0
        # 2.) Iterate through num. of iterations
        for iteration in range(self.num_iterations):
            # 3.) Iteration through m training examples
            for xi, yi in zip(X, y):
                # 4.) Linear combiner
                z = np.dot(xi, self.w) + self.b
                # 5.) Step unit activation function
                yi_pred = self._unit_step_function(z)
                # 6.) Update the weights
                delta = self.lr * (yi - yi_pred)
                self.w += delta * xi
                self.b += delta
                     
    def predict(self, X):
        """ 
        Parameters:
        X of shape (m, n_x) or xi of shape (1, n_x)
            
        """
        z = np.dot(X, self.w) + self.b
        predictions = self._unit_step_function(z)
        return predictions
    
    def accuracy(self, y_pred, y):
        """ 
        Parameters:
        y_pred of shape (m,) or (m, 1)
        y test of shape (m,) or (m, 1) 
        """
        return np.mean(np.squeeze(y_pred) == np.squeeze(y)) * 100
        
    def _unit_step_function(self, z):
        return np.where(z>0, 1, 0)
    
#%% Dataset I (IRIS)

iris = datasets.load_iris()
X = np.array(iris.data[:100,0:2])
y = iris.target[:100]
del iris

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

#%% Dataset II (BLOB)

X, y = datasets.make_blobs(n_samples=150,n_features=2,centers=2,cluster_std=1.05,random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#%% Predict
    
model = Perceptron()    
model.fit(X_train, y_train)    
predictions = model.predict(X_test)    
accuracy = model.accuracy(predictions, y_test)

#%% Visualize the results

w = model.w
b = model.b

# predict 1 if (w[0]*X_train[:,0] + w[1]*X_train[:,1] + b) > 0
# predict 0 if (w[0]*X_train[:,0] + w[1]*X_train[:,1] + b) <= 0

x0_1 = np.amin(X_train[:,0])
x0_2 = np.amax(X_train[:,0])
x1_1 = (-w[0] * x0_1 - b) / w[1]
x1_2 = (-w[0] * x0_2 - b) / w[1]
plt.plot([x0_1, x0_2],[x1_1, x1_2])
plt.scatter(X_train[:,0], X_train[:,1], c=y_train)
