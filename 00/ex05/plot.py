from prediction import predict_
import numpy as np
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')

# For my windows

def plot(x, y, theta):
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
	plt.scatter(x, y, color="red")
	plt.plot(x, predict_(x, theta), color="blue")
	plt.show()
	return None
