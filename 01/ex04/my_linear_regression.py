import numpy as np
# import matplotlib.pyplot as plt

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
		# if type(x).__module__ != np.__name__ or type(y).__module__ != np.__name__ or type(self.thetas).__module__ != np.__name__:
		# 	return None
		# if self.thetas.shape != (2, 1) or y.shape[1] != 1 or x.shape[1] != 1 or x.shape != y.shape:
		# 	return None

		x_prime = add_intercept(x)
		x_prime_t = np.transpose(x_prime)
		thetas_history = []
		j_history = []
		for i in range(0, self.max_iter):
			tb = self.thetas
			y_pred = np.dot(x_prime, tb)
			# dot = np.dot(x_prime, tb)
			j = np.dot(x_prime_t, (y_pred - y)) / len(x)
			self.thetas = tb - self.alpha * j
			thetas_history.append(tb[1])
			j_history.append(self.loss_(y, y_pred))
		# self.plot_loss_(thetas_history, j_history)
		return self.thetas

	def plot_loss_(self, thetas_history, j_history):
		f = plt.figure("are_blue_pills_magics")
		f.clear()
		plt.plot(thetas_history, j_history, color="grey", linestyle="--", label="test")
		plt.legend()
		plt.xlabel("Theta 1")
		plt.ylabel("Loss")
		#plt.plot(self.X, Y_pred, color="red")
		#plt.scatter(self.X, self.Y)
		plt.ioff()
		plt.show()

	def predict_(self, x):
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
