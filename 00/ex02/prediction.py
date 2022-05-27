import numpy as np

def simple_predict(x, theta):
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
	print(x.shape, theta.shape)
	if type(x).__module__ != np.__name__ or type(theta).__module__ != np.__name__ or \
	x.shape[1] != 1 or theta.shape != (2, 1) or len(x) == 0:
		return None
	y_pred = []
	for i in range(0, len(x)):
		y_pred.append([float(x[i][0] * theta[1][0] + theta[0][0])])
	return y_pred
