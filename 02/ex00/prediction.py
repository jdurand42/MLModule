import numpy as np

def simple_predict(x, theta):
	"""Computes the prediction vector y_hat from two non-empty numpy.array.
	Args:
		x: has to be an numpy.array, a matrix of dimension m * n.
		theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.
	Return:
		y_hat as a numpy.array, a vector of dimension m * 1.
		None if x or theta are empty numpy.array.
		None if x or theta dimensions are not matching.
		None if x or theta is not of expected type.
	Raises:
		This function should not raise any Exception.
	"""
	if type(x).__module__ != np.__name__ or type(theta).__module__ != np.__name__:
		return None
	if len(x) == 0 or len(theta) == 0:
		return None
	if theta.shape[1] != 1:
		return None
	if x.shape[1] + 1 != theta.shape[0]:
		return None

	y_pred = []
	for i in range(0, len(x)):
		res = 0
		res += theta[0]
		for j in range(0, len(x[i])):
			res += (theta[j + 1] * x[i][j])
		y_pred.append(res)
	return np.array(y_pred)
