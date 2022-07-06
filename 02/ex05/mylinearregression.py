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

class MyLinearRegression():
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

		x_prime = add_intercept(x)
		x_prime_t = np.transpose(x_prime)
		for i in range(0, self.max_iter):
			y_pred = np.dot(x_prime, self.thetas)
			j = (np.dot(x_prime_t, y_pred - y)) / len(x)
			self.thetas = self.thetas - self.alpha * j
		return self.thetas

	def predict_(self, x):
		if type(x).__module__ != np.__name__ or type(self.thetas).__module__ != np.__name__:
			return None
		if len(x) == 0 or len(self.thetas) == 0:
			return None
		if self.thetas.shape[1] != 1:
			return None
		if x.shape[1] + 1 != self.thetas.shape[0]:
			return None
		x = add_intercept(x)
		y_pred = np.dot(x, self.thetas)
		return y_pred

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
		if type(y).__module__ != np.__name__ or type(y_hat).__module__ != np.__name__ \
		or y_hat.shape != y.shape:
			return None
		return (y_hat - y) * (y_hat - y)

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
		return self.loss_elem_(y, y_hat).mean() / 2
