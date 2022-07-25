import numpy as np

def log_loss_(y, y_hat, eps=1e-15):
	"""
	Computes the logistic loss value.
	Args:
		y: has to be an numpy.ndarray, a vector of shape m * 1.
		y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
		eps: has to be a float, epsilon (default=1e-15)
	Returns:
		The logistic loss value as a float.
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	if type(y).__module__ != np.__name__:
		return None
	if type(y_hat).__module__ != np.__name__:
		return None
	if isinstance(eps, float) == False:
		return None
	if len(y) == 0 or len(y_hat) == 0:
		return None
	if y.shape[1] != 1 or y.shape != y_hat.shape:
		return None

	j = (y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
	j = -j.mean()
	return j

def vec_log_loss_(y, y_hat, eps=1e-15):
	"""
	Computes the logistic loss value.
	Args:
		y: has to be an numpy.ndarray, a vector of shape m * 1.
		y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
		eps: has to be a float, epsilon (default=1e-15)
	Returns:
		The logistic loss value as a float.
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	if type(y).__module__ != np.__name__:
		return None
	if type(y_hat).__module__ != np.__name__:
		return None
	if isinstance(eps, float) == False:
		return None
	if len(y) == 0 or len(y_hat) == 0:
		return None
	if y.shape[1] != 1 or y.shape != y_hat.shape:
		return None

	ones = np.ones(y.shape)
	j = (np.dot(np.transpose(y), np.log(y_hat + eps)) + \
	(np.dot(np.transpose(ones - y), np.log(ones - y_hat + eps))))
	j = j / -len(y)

	return j.mean()

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

def reg_log_loss_(y, y_hat, theta, lambda_):
	"""Computes the regularized loss of a logistic regression model from two non-empty numpy.ndarray, without any for l
	Args:
		y: has to be an numpy.ndarray, a vector of shape m * 1.
		y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
		theta: has to be a numpy.ndarray, a vector of shape n * 1.
		lambda_: has to be a float.
	Returns:
		The regularized loss as a float.
		None if y, y_hat, or theta is empty numpy.ndarray.
		None if y and y_hat do not share the same shapes.
	Raises:
		This function should not raise any Exception.
	"""
	if type(y).__module__ != np.__name__:
		return None
	if type(y_hat).__module__ != np.__name__:
		return None
	if type(theta).__module__ != np.__name__:
		return None
	if len(theta) == 0 or theta.shape[1] != 1:
		return None
	if len(y) == 0 or y.shape[1] != 1:
		return None
	if y.shape != y_hat.shape:
		return None
	if isinstance(lambda_, float) == False:
		return None
	return vec_log_loss_(y, y_hat) + (lambda_ * l2(theta) /( 2 * len(y)))
