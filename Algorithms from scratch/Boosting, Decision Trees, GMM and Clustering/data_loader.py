import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def toy_data_1():
	features = [[1.,1.], [-1., 1.], [-1.,-1.], [1.,-1.]]
	labels = [1, -1,  1, -1]
	return features, labels

def toy_data_2():
	features = [[0., 1.414], [-1.414, 0.], [0., -1.414], [1.414, 0.]]
	labels = [1, -1, 1, -1]
	return features, labels

def toy_data_3():
	features = [[1,2], [2,1], [2,3], [3,2]]
	labels = [0, 0, 1, 1]
	return features, labels


def binary_iris_dataset():
	iris = load_iris()
	X = iris.data[50: , ]
	y = iris.target[50: , ]
	y = y * 2 - 3

	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=3)
	return X_train.tolist(), X_test.tolist(), y_train.tolist(), y_test.tolist()

def discrete_2D_iris_dataset():
	iris = load_iris()
	X = iris.data[:, [0,1]]
	y = iris.target

	X_discrete = np.ones(X.shape)
	X_discrete[X[:,0]<5.45, 0] = 0
	X_discrete[X[:,0]>=6.15, 0] = 2
	X_discrete[X[:,1]<2.8, 1] = 0
	X_discrete[X[:,1]>=3.45, 1] = 2

	X_train, X_test, y_train, y_test = train_test_split(X_discrete, y, train_size=0.8, random_state=3)
	return X_train.tolist(), X_test.tolist(), y_train.tolist(), y_test.tolist()

def toy_dataset(cluster_size=2, sample_per_cluster=50):
    np.random.seed(42)
    N = cluster_size*sample_per_cluster
    y = np.zeros(N)
    x = np.random.standard_normal(size=(N, 2))
    for i in range(cluster_size):
        theta = 2*np.pi*i/cluster_size
        x[i*sample_per_cluster:(i+1)*sample_per_cluster] = x[i*sample_per_cluster:(i+1)*sample_per_cluster] + \
            (cluster_size*np.cos(theta), cluster_size*np.sin(theta))
        y[i*sample_per_cluster:(i+1)*sample_per_cluster] = i
    return x, y


def load_digits():
    digits = datasets.load_digits()
    x = digits.data/16
    x = x.reshape([x.shape[0], -1])
    y = digits.target
    return train_test_split(x, y, random_state=42, test_size=0.25)