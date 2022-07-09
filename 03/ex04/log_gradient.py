import numpy as np
from log_pred import logistic_predict_
# from vec_log_loss_ import vec_log_loss_

def log_gradient(x, y, theta):
	"""Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop. The three arrays must have compatibl
	Args:
		x: has to be an numpy.ndarray, a matrix of shape m * n.
		y: has to be an numpy.ndarray, a vector of shape m * 1.
		theta: has to be an numpy.ndarray, a vector of shape (n + 1) * 1.
	Returns:
		The gradient as a numpy.ndarray, a vector of shape n * 1, containing the result of the formula for all j.
		None if x, y, or theta are empty numpy.ndarray.
		None if x, y and theta do not have compatible dimensions.
	Raises:
		This function should not raise any Exception.
	"""
	if type(x).__module__ != np.__name__ or type(theta).__module__ != np.__name__ \
	or type(y).__module__ != np.__name__:
		return None
	if len(x) == 0 or len(theta) == 0 or len(y) == 0:
		return None
	if theta.shape[1] != 1:
		return None
	if x.shape[1] + 1 != theta.shape[0]:
		return None
	if y.shape[1] != 1 or y.shape[0] != x.shape[0]:
		return None

	y_pred = logistic_predict_(x, theta)
	j = np.zeros(theta.shape, dtype="float64")
	print(x)
	j[0][0] = (y_pred - y).sum() / len(y)
	for i in range(1, len(theta)):
		j[i][0] = ((y_pred - y) * x[:,[i-1]]).sum() / len(y)
	return j
