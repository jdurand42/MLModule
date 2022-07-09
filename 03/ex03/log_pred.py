import numpy as np

def sigmoid_(x):
	"""
	Compute the sigmoid of a vector.
	Args:
		x: has to be a numpy.ndarray of shape (m, 1).
	Returns:
		The sigmoid value as a numpy.ndarray of shape (m, 1).
		None if x is an empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""
	if type(x).__module__ != np.__name__:
		return None
	if len(x) == 0 or x.shape[1] != 1:
		return None

	sig = 1 / (1 + np.exp(-x))
	return sig

def add_intercept(x):
	"""Adds a column of 1’s to the non-empty numpy.array x.
	Args:
		x: has to be an numpy.array, a vector of shape m * n.
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

def logistic_predict_(x, theta):
	"""Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
	Args:
		x: has to be an numpy.ndarray, a vector of dimension m * n.
		theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
	Returns:
		y_hat as a numpy.ndarray, a vector of dimension m * 1.
		None if x or theta are empty numpy.ndarray.
		None if x or theta dimensions are not appropriate.
	Raises:
		This function should not raise any Exception.
	"""
	if type(x).__module__ != np.__name__:
		return None
	if type(theta).__module__ != np.__name__:
		return None
	if len(x) == 0:
		return None
	if len(theta) == 0 or theta.shape[0] != x.shape[1] + 1 or theta.shape[1] != 1:
		return None

	x_prime = add_intercept(x)
	return sigmoid_(np.dot(x_prime, theta))
