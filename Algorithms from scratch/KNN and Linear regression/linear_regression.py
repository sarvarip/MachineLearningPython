"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, regularized_linear_regression,
tune_lambda, and test_error.
"""

import numpy as np
import pandas as pd


###### Q1.1 ######
def linear_regression_noreg(X, y):
	"""
	Compute the weight parameter given X and y.
	Inputs:
	- X: A numpy array of shape (num_samples, D) containing feature.
	- y: A numpy array of shape (num_samples, ) containing label
	Returns:
	- w: a numpy array of shape (D, )
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################

	w = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)), y)

	return w


###### Q1.2 ######
def regularized_linear_regression(X, y, lambd):
	"""
		Compute the weight parameter given X, y and lambda.
		Inputs:
		- X: A numpy array of shape (num_samples, D) containing feature.
		- y: A numpy array of shape (num_samples, ) containing label
		- lambd: a float number containing regularization strength
		Returns:
		- w: a numpy array of shape (D, )
		"""
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################

	w = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X) + lambd*np.identity(np.size(X, 1))), np.transpose(X)), y)

	return w


###### Q1.3 ######
def tune_lambda(Xtrain, ytrain, Xval, yval, lambds):
	"""
		Find the best lambda value.
		Inputs:
		- Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
		- ytrain: A numpy array of shape (num_training_samples, ) containing training label
		- Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
		- yval: A numpy array of shape (num_val_samples, ) containing validation label
		- lambds: a list of lambdas
		Returns:
		- bestlambda: the best lambda you find in lambds
		"""
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################

	err = [0] * len(lambds)
	i = 0
	for lambd in lambds:
		w = regularized_linear_regression(Xtrain, ytrain, lambd)
		err[i] = test_error(w, Xval, yval)
		i = i + 1
	bestlambda = np.argmin(err)
	return bestlambda


###### Q1.4 ######
def test_error(w, X, y):
	"""
		Compute the mean squre error on test set given X, y, and model parameter w.
		Inputs:
		- X: A numpy array of shape (num_samples, D) containing test feature.
		- y: A numpy array of shape (num_samples, ) containing test label
		- w: a numpy array of shape (D, )
		Returns:
		- err: the mean square error
		"""

	ypred = np.matmul(X,w)
	err = np.mean(np.square(ypred-y))

	return err


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

def data_processing():
	white = pd.read_csv('winequality-white.csv', low_memory=False, sep=';').values

	[N, d] = white.shape

	np.random.seed(3)
	# prepare data
	ridx = np.random.permutation(N)
	ntr = int(np.round(N * 0.8))
	nval = int(np.round(N * 0.1))
	ntest = N - ntr - nval

	# spliting training, validation, and test

	Xtrain = np.hstack([np.ones([ntr, 1]), white[ridx[0:ntr], 0:-1]])

	ytrain = white[ridx[0:ntr], -1]

	Xval = np.hstack([np.ones([nval, 1]), white[ridx[ntr:ntr + nval], 0:-1]])
	yval = white[ridx[ntr:ntr + nval], -1]

	Xtest = np.hstack([np.ones([ntest, 1]), white[ridx[ntr + nval:], 0:-1]])
	ytest = white[ridx[ntr + nval:], -1]
	return Xtrain, ytrain, Xval, yval, Xtest, ytest


def main():
	np.set_printoptions(precision=3)
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing()
	# =========================Q 1.1 linear_regression=================================
	w = linear_regression_noreg(Xtrain, ytrain)
	print("======== Question 1.1 Linear Regression ========")
	print("dimensionality of the model parameter is ", len(w), ".", sep="")
	print("model parameter is ", np.array_str(w))

	# =========================Q 1.2 regularized linear_regression=====================
	lambd = 5.0
	wl = regularized_linear_regression(Xtrain, ytrain, lambd)
	print("\n")
	print("======== Question 1.2 Regularized Linear Regression ========")
	print("dimensionality of the model parameter is ", len(wl), sep="")
	print("lambda = ", lambd, ", model parameter is ", np.array_str(wl), sep="")

	# =========================Q 1.3 tuning lambda======================
	lambds = [0, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1, 10, 10 ** 2]
	bestlambd = tune_lambda(Xtrain, ytrain, Xval, yval, lambds)
	print("\n")
	print("======== Question 1.3 tuning lambdas ========")
	print("tuning lambda, the best lambda =  ", bestlambd, sep="")

	# =========================Q 1.4 report mse on test ======================
	wbest = regularized_linear_regression(Xtrain, ytrain, bestlambd)
	mse = test_error(wbest, Xtest, ytest)
	print("\n")
	print("======== Question 1.4 report MSE ========")
	print("MSE on test is %.3f" % mse)

if __name__ == "__main__":
		main()
