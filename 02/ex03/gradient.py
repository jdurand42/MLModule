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

def gradient(x, y, theta):
	"""Computes a gradient vector from three non-empty numpy.array, without any for-loop.
	The three arrays must have the compatible dimensions.
	Args:
		x: has to be an numpy.array, a matrix of dimension m * n.
		y: has to be an numpy.array, a vector of dimension m * 1.
		theta: has to be an numpy.array, a vector (n +1) * 1.
	Return:
		The gradient as a numpy.array, a vector of dimensions n * 1,
		containg the result of the formula for all j.
		None if x, y, or theta are empty numpy.array.
		None if x, y and theta do not have compatible dimensions.
		None if x, y or theta is not of expected type.
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

	x_prime = add_intercept(x)
	x_prime_t = np.transpose(x_prime)
	y_pred = np.dot(x_prime, theta)
	# print(y_pred)
	j = (np.dot(x_prime_t, y_pred - y)) / len(x)
	return j
