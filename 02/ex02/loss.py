import numpy as np

def loss_elem_(y, y_hat):
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

def loss_(y, y_hat):
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
	return loss_elem_(y, y_hat).mean() / 2
