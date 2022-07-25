import numpy as np

def add_polynomial_features(x, power):
	"""Add polynomial features to matrix x by raising its columns to every power in the range of 1 up to the power give
	Args:
		x: has to be an numpy.ndarray, a matrix of shape m * n.
		power: has to be an int, the power up to which the columns of matrix x are going to be raised.
	Returns:
		The matrix of polynomial features as a numpy.ndarray, of shape m * (np), containg the polynomial feature va
		None if x is an empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""
	try:
		res = []
		for i in range(0, len(x)):
			res.append([])
			for pow in range(1, power+1):
				for j in range(0, len(x[i])):
					res[i].append(x[i][j] ** pow)
		return np.array(res)
	except:
		return None
