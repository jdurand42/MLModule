import numpy as np
from math import sqrt

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


def vec_log_gradient(x, y, theta):
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
	x_prime = add_intercept(x)
	j = (np.dot(np.transpose(x_prime), y_pred - y)) / len(y)
	return j

def sigmoid_(x):
	"""
	Compute the sigmoid of a vector.
	Args:
		x: has to be a numpy.ndarray of shape (m, 1).
	Returns:
		The sigmoid value as a numpy.ndarray of shape (m, 1).
		None if x is an empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""
	if type(x).__module__ != np.__name__:
		return None
	if len(x) == 0 or x.shape[1] != 1:
		return None

	sig = 1 / (1 + np.exp(-x))
	return sig

class MyLogisticRegression():
	def __init__(self, thetas, alpha=0.001, max_iter=1000):
		self.alpha = alpha
		self.thetas = thetas
		self.max_iter = max_iter

	def fit_(self, x, y):
		if type(x).__module__ != np.__name__ or type(self.thetas).__module__ != np.__name__:
			return None
		if type(y).__module__ != np.__name__:
			return None
		if len(x) == 0 or len(self.thetas) == 0 or len(y) == 0:
			return None
		if self.thetas.shape[1] != 1:
			return None
		if x.shape[1] + 1 != self.thetas.shape[0]:
			return None
		if y.shape[1] != 1 or y.shape[0] != x.shape[0]:
			return None
		if isinstance(self.alpha, float) == False or isinstance(self.max_iter, int) == False:
			return None

		for i in range(0, self.max_iter):
			y_pred = self.predict_(x)
			x_prime = add_intercept(x)
			j =  (np.dot(np.transpose(x_prime), y_pred - y)) / len(y)
			self.thetas = self.thetas - self.alpha * j
		return self.thetas

	def predict_(self, x):
		if type(x).__module__ != np.__name__:
			return None
		if type(self.thetas).__module__ != np.__name__:
			return None
		if len(x) == 0:
			return None
		if len(self.thetas) == 0 or self.thetas.shape[0] != x.shape[1] + 1 or self.thetas.shape[1] != 1:
			return None

		x_prime = add_intercept(x)
		return sigmoid_(np.dot(x_prime, self.thetas))

	def loss_elem_(self, y, y_hat):
		"""
		Description:
			Calculates all the elements (y_pred - y)^2 of the loss function.
		Args:
			y: has to be an numpy.array, a vector.
			y_hat: has to be an numpy.array, a vector.
		Returns:
			J_elem: numpy.array, a vector of dimension (number of the training examples,1).
			None if there is a dimension matching problem between y and y_hat.
			None if y or y_hat is not of the expected type.
		Raises:
			This function should not raise any Exception.
		"""
		if type(y).__module__ != np.__name__:
			return None
		if type(y_hat).__module__ != np.__name__:
			return None
		if len(y) == 0 or len(y_hat) == 0:
			return None
		if y.shape[1] != 1 or y.shape != y_hat.shape:
			return None

		eps = 1e-15
		j = (y * np.log(y_hat + eps) + (1 - y) * np.log(1 - (y_hat + eps)))
		j = -j
		return j


	def loss_(self, y, y_hat):
		"""
		Description:
			Calculates the value of loss function.
		Args:
			y: has to be an numpy.array, a vector.
			y_hat: has to be an numpy.array, a vector.
		Returns:
			J_value : has to be a float.
			None if there is a shape matching problem between y or y_hat.
			None if y or y_hat is not of the expected type.
		Raises:
			This function should not raise any Exception.
		"""
		if type(y).__module__ != np.__name__:
			return None
		if type(y_hat).__module__ != np.__name__:
			return None
		if len(y) == 0 or len(y_hat) == 0:
			return None
		if y.shape[1] != 1 or y.shape != y_hat.shape:
			return None
		eps = 1e-15
		ones = np.ones(y.shape)
		j = (np.dot(np.transpose(y), np.log(y_hat + eps)) + \
		(np.dot(np.transpose(ones - y), np.log(ones - y_hat + eps))))

		j = j / -len(y)
		return j.mean()

	# def score(self, y, y_hat):
	# 	"""
	# 		gives r2 score
	# 	"""
	# 	r2 = 1 - (((y - y_hat) * (y - y_hat)).sum() / (((y - y.mean()) * (y - y.mean())).sum()))
	# 	return r2
