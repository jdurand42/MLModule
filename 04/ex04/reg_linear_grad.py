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

def predict_(x, theta):
	x = add_intercept(x)
	y_pred = np.dot(x, theta)
	return y_pred

def reg_linear_grad(y, x, theta, lambda_):
	"""Computes the regularized linear gradient of three non-empty numpy.ndarray,
	with two for-loop. The three arrays must have compatible shapes.
	Args:
		y: has to be a numpy.ndarray, a vector of shape m * 1.
		x: has to be a numpy.ndarray, a matrix of dimesion m * n.
		theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
		lambda_: has to be a float.
	Return:
		A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
		None if y, x, or theta are empty numpy.ndarray.
		None if y, x or theta does not share compatibles shapes.
		None if y, x or theta or lambda_ is not of the expected type.
	Raises:
		This function should not raise any Exception.
	"""
	if type(y).__module__ != np.__name__:
		return None
	if type(x).__module__ != np.__name__:
		return None
	if type(theta).__module__ != np.__name__:
		return None
	if isinstance(lambda_, (float, int)) == False:
		return None
	if len(y) == 0 or len(x) == 0 or len(theta) == 0:
		return None
	if y.shape[1] != 1 or theta.shape[1] != 1:
		return None
	if x.shape[1] + 1 != theta.shape[0]:
		return None
	if x.shape[0] != y.shape[0]:
		return None


	y_pred = predict_(x, theta)
	thetas = np.zeros(theta.shape)
	j = (y_pred - y).sum() / len(y)
	thetas[0][0] = j
	for i in range(1, len(theta)):
		j = ((y_pred - y) * x[:,[i-1]]).sum()
		j += lambda_ * (theta[i][0])
		j = j / len(y)
		thetas[i][0] = j
	return np.array(thetas)

def vec_reg_linear_grad(y, x, theta, lambda_):
	"""Computes the regularized linear gradient of three non-empty numpy.ndarray,
	without any for-loop. The three arrays must have compatible shapes.
	Args:
		y: has to be a numpy.ndarray, a vector of shape m * 1.
		x: has to be a numpy.ndarray, a matrix of dimesion m * n.
		theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
		lambda_: has to be a float.
	Return:
		A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
		None if y, x, or theta are empty numpy.ndarray.
		None if y, x or theta does not share compatibles shapes.
		None if y, x or theta or lambda_ is not of the expected type.
	Raises:
		This function should not raise any Exception.
	"""
	if type(y).__module__ != np.__name__:
		return None
	if type(x).__module__ != np.__name__:
		return None
	if type(theta).__module__ != np.__name__:
		return None
	if isinstance(lambda_, (float, int)) == False:
		return None
	if len(y) == 0 or len(x) == 0 or len(theta) == 0:
		return None
	if y.shape[1] != 1 or theta.shape[1] != 1:
		return None
	if x.shape[1] + 1 != theta.shape[0]:
		return None
	if x.shape[0] != y.shape[0]:
		return None
	x_prime = add_intercept(x)
	y_pred = predict_(x, theta)
	theta = theta.copy()
	theta[0][0] = 0
	j = (np.dot(np.transpose(x_prime), y_pred - y) + lambda_ * theta) / len(y)
	return j
