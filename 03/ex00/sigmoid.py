import numpy as np

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
