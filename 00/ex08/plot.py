from prediction import predict_
import numpy as np
import matplotlib.pyplot as plt
from vec_loss import loss_
# matplotlib.use('TkAgg')

# For my windows

def plot_with_loss(x, y, theta):
	"""Plot the data and prediction line from three non-empty numpy.array.
	Args:
		x: has to be an numpy.array, a vector of shape m * 1.
		y: has to be an numpy.array, a vector of shape m * 1.
		theta: has to be an numpy.array, a vector of shape 2 * 1.
	Returns:
		Nothing.
	Raises:
		This function should not raise any Exception.
	"""
	if type(x).__module__ != np.__name__ or type(theta).__module__ != np.__name__ or \
	type(y).__module__ != np.__name__ or y.shape[1] != 1 or y.shape != x.shape or \
	x.shape[1] != 1 or theta.shape != (2, 1) or len(x) == 0:
		return None
	plt.scatter(x, y, color="blue")
	y_pred = predict_(x, theta)
	plt.plot(x, y_pred, color="orange")
	# loss_elem = loss_elem_(x, y_pred)
	for i in range(0, len(x)):
		plt.plot([x[i], x[i]], [y[i], y_pred[i]], color="red", linestyle="--")
	cost = loss_(y, y_pred)
	plt.title(f"Cost: {cost * 2}")
	plt.show()
	return None
