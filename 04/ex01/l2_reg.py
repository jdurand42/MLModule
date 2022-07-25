import numpy as np

def iterative_l2(theta):
	"""Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
	Args:
		theta: has to be a numpy.ndarray, a vector of shape n * 1.
	Returns:
		The L2 regularization as a float.
		None if theta in an empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""
	if type(theta).__module__ != np.__name__:
		return None
	if len(theta) == 0 or theta.shape[1] != 1:
		return None
	sum = 0
	for i in range(1, len(theta)):
		sum += (theta[i][0] ** 2)
	return sum

def l2(theta):
	"""Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
	Args:
		theta: has to be a numpy.ndarray, a vector of shape n * 1.
	Returns:
		The L2 regularization as a float.
		None if theta in an empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""
	if type(theta).__module__ != np.__name__:
		return None
	if len(theta) == 0 or theta.shape[1] != 1:
		return None
	theta[0][0] = 0
	return (theta * theta).sum()
