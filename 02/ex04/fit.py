import numpy as np

def add_intercept(x):
	"""Adds a column of 1’s to the non-empty numpy.array x.
	Args:
		x: has to be an numpy.array, a vector of shape m * 1.
	Returns:
		x as a numpy.array, a vector of shape m * 2.
		None if x is not a numpy.array.
		None if x is a empty numpy.array.
	Raises:
	This function should not raise any Exception.
	"""
	if type(x).__module__ != np.__name__ or len(x) == 0:
		return None
	return np.insert(x, 0, np.ones(x.shape[0],), axis=1)

def fit_(x, y, theta, alpha, max_iter):
	"""
	Description:
		Fits the model to the training dataset contained in x and y.
	Args:
		x: has to be a numpy.array, a matrix of dimension m * n:
		(number of training examples, number of features).
		y: has to be a numpy.array, a vector of dimension m * 1:
		(number of training examples, 1).
		theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
		(number of features + 1, 1).
		alpha: has to be a float, the learning rate
		max_iter: has to be an int, the number of iterations done during the gradient descent
	Return:
		new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
		None if there is a matching dimension problem.
		None if x, y, theta, alpha or max_iter is not of expected type.
	Raises:
		This function should not raise any Exception.
	"""

	if type(x).__module__ != np.__name__ or type(theta).__module__ != np.__name__:
		return None
	if type(y).__module__ != np.__name__:
		return None
	if len(x) == 0 or len(theta) == 0 or len(y) == 0:
		return None
	if theta.shape[1] != 1:
		return None
	if x.shape[1] + 1 != theta.shape[0]:
		return None
	if y.shape[1] != 1 or y.shape[0] != x.shape[0]:
		return None
	if isinstance(alpha, float) == False or isinstance(max_iter, int) == False:
		return None

	x_prime = add_intercept(x)
	x_prime_t = np.transpose(x_prime)
	for i in range(0, max_iter):
		y_pred = np.dot(x_prime, theta)
		j = (np.dot(x_prime_t, y_pred - y)) / len(x)
		theta = theta - alpha * j
	return theta
