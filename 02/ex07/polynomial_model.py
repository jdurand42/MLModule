import numpy as np

def add_polynomial_features(x, power):
	"""Add polynomial features to vector x by raising its values up to the power given in argument.
	Args:
		x: has to be an numpy.array, a vector of dimension m * 1.
		power: has to be an int, the power up to which the components of vector x are going to be raised.
	Return:
		The matrix of polynomial features as a numpy.array, of dimension m * n,
		containing the polynomial feature values for all training examples.
		None if x is an empty numpy.array.
		None if x or power is not of expected type.
	Raises:
		This function should not raise any Exception.
	"""
	if type(x).__module__ != np.__name__ or isinstance(power, int) == False:
		return None
	if len(x) == 0 or x.shape[1] != 1:
		return None
	res = []
	for i in range(0, len(x)):
		res.append([x[i][0]])
		for j in range(2, power + 1):
			res[i].append(x[i][0] ** j)
	return np.array(res)
