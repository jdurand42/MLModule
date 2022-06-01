from math import sqrt
import numpy as np

def mse_(y, y_hat):
	"""
	Description:
	 Calculate the MSE between the predicted output and the real output.
	Args:
	 y: has to be a numpy.array, a vector of shape m * 1.
	 y_hat: has to be a numpy.array, a vector of shape m * 1.
	Returns:
	 mse: has to be a float.
	 None if there is a matching shape problem.
	Raises:
	 This function should not raise any Exception.
	 """
	if type(y).__module__ != np.__name__ or type(y_hat).__module__ != np.__name__ \
 	or y_hat.shape != y.shape or y.shape[1] != 1:
 		return None
	met = ((y_hat - y) * (y_hat - y)).mean()
	return met

def rmse_(y, y_hat):
	"""
	Description:
		Calculate the RMSE between the predicted output and the real output.
	Args:
		y: has to be a numpy.array, a vector of shape m * 1.
		y_hat: has to be a numpy.array, a vector of shape m * 1.
	Returns:
		rmse: has to be a float.
		None if there is a matching shape problem.
	Raises:
		This function should not raise any Exception.
	"""
	if type(y).__module__ != np.__name__ or type(y_hat).__module__ != np.__name__ \
 	or y_hat.shape != y.shape or y.shape[1] != 1:
 		return None
	return sqrt(mse_(y, y_hat))

def mae_(y, y_hat):
	"""
	Description:
		Calculate the MAE between the predicted output and the real output.
	Args:
		y: has to be a numpy.array, a vector of shape m * 1.
		y_hat: has to be a numpy.array, a vector of shape m * 1.
	Returns:
		mae: has to be a float.
		None if there is a matching shape problem.
	Raises:
		This function should not raise any Exception.
	"""
	if type(y).__module__ != np.__name__ or type(y_hat).__module__ != np.__name__ \
 	or y_hat.shape != y.shape or y.shape[1] != 1:
 		return None
	return np.absolute(y_hat - y).mean()

def r2score_(y, y_hat):
	"""
	Description:
		Calculate the R2score between the predicted output and the output.
	Args:
		y: has to be a numpy.array, a vector of shape m * 1.
		y_hat: has to be a numpy.array, a vector of shape m * 1.
	Returns:
		r2score: has to be a float.
		None if there is a matching shape problem.
	Raises:
		This function should not raise any Exception.
	"""
	if type(y).__module__ != np.__name__ or type(y_hat).__module__ != np.__name__ \
 	or y_hat.shape != y.shape or y.shape[1] != 1:
 		return None
	return 1 - (((y_hat - y) * (y_hat - y)).sum() / ((y - y.mean()) * (y - y.mean())).sum())
