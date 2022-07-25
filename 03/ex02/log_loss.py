import numpy as np
# from log_pred import logistic_predict_

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
