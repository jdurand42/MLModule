# what about empty array
# check of non float ?
import numpy as np

def loss_(y, y_hat):
	"""Computes the half mean squared error of two non-empty numpy.array, without any for loop.
		The two arrays must have the same dimensions.
	Args:
		y: has to be an numpy.array, a vector.
		y_hat: has to be an numpy.array, a vector.
	Returns:
		The half mean squared error of the two vectors as a float.
		None if y or y_hat are empty numpy.array.
		None if y and y_hat does not share the same dimensions.
		None if y or y_hat is not of the expected type.
	Raises:
		This function should not raise any Exception.
	"""
	if type(y).__module__ != np.__name__ or type(y_hat).__module__ != np.__name__ \
	or y_hat.shape != y.shape:
		return None
	return ((y_hat - y) * (y_hat - y)).mean() / 2
