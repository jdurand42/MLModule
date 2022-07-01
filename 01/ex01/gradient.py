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
#
# def predict(x, y, theta):
# 	x = add_intercept(x)
# 	y_pred = np.dot(x, theta)
# 	return y_pred

def gradient(x, y, theta):
	"""Computes a gradient vector from three non-empty numpy.array, without any for-loop.
	The three arrays must have compatible shapes.
	Args:
		x: has to be an numpy.array, a vector of shape m * 1.
		y: has to be an numpy.array, a vector of shape m * 1.
		theta: has to be an numpy.array, a 2 * 1 vector.
	Return:
		The gradient as a numpy.array, a vector of shape 2 * 1.
		None if x, y, or theta are empty numpy.array.
		None if x, y and theta do not have compatible shapes.
		None if x, y or theta is not of the expected type.
	Raises:
		This function should not raise any Exception.
	"""
	if type(x).__module__ != np.__name__ or type(y).__module__ != np.__name__ or type(theta).__module__ != np.__name__:
		return None
	if theta.shape != (2, 1) or y.shape[1] != 1 or x.shape[1] != 1 or x.shape != y.shape:
		return None
	# y_pred = predict(x, y, theta)
	# j0 = (y_pred - y).sum() / len(y)
	# j1 = ((y_pred - y) * x).sum() / len(y)
	x_prime = add_intercept(x)
	x_prime_t = np.transpose(x_prime)
	dot = np.dot(x_prime, theta)
	j = np.dot(x_prime_t, (dot - y)) / len(x)
	return j
