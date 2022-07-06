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

def simple_predict(x, theta):
	"""Computes the prediction vector y_hat from two non-empty numpy.array.
	Args:
		x: has to be an numpy.array, a matrix of dimension m * n.
		theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.
	Return:
		y_hat as a numpy.array, a vector of dimension m * 1.
		None if x or theta are empty numpy.array.
		None if x or theta dimensions are not matching.
		None if x or theta is not of expected type.
	Raises:
		This function should not raise any Exception.
	"""
	if type(x).__module__ != np.__name__ or type(theta).__module__ != np.__name__:
		return None
	if len(x) == 0 or len(theta) == 0:
		return None
	if theta.shape[1] != 1:
		return None
	if x.shape[1] + 1 != theta.shape[0]:
		return None

	x_prime = add_intercept(x)
	print(x_prime)
	y_pred = np.dot(x_prime, theta)

	return y_pred
