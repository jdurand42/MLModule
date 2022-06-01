import numpy as np
from tools import add_intercept

def predict_(x, theta):
	#  NOTE verifier check error
	"""Computes the vector of prediction y_hat from two non-empty numpy.array.
	Args:
		x: has to be an numpy.array, a vector of shape m * 1.
		theta: has to be an numpy.array, a vector of shape 2 * 1.
	Returns:
		y_hat as a numpy.array, a vector of shape m * 1.
		None if x or theta are empty numpy.array.
		None if x or theta shapes are not appropriate.
		None if x or theta is not of the expected type.
	Raises:
		This function should not raise any Exception.
	"""
	# print(x.shape, theta.shape)
	if type(x).__module__ != np.__name__ or type(theta).__module__ != np.__name__ or \
	x.shape[1] != 1 or theta.shape != (2, 1) or len(x) == 0:
		return None

	x = add_intercept(x)
	y_pred = np.dot(x, theta)
	return y_pred
