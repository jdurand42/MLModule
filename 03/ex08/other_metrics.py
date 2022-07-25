import numpy as np

def accuracy_score_(y, y_hat):
	"""
		Compute the accuracy score.
	Args:
		y:a numpy.ndarray for the correct labels
		y_hat:a numpy.ndarray for the predicted labels
	Returns:
		The accuracy score as a float.
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	try:
		true = 0
		for i in range(0, len(y)):
			if y_hat[i][0] == y[i][0]:
				true += 1
		return true / len(y)
	except:
		return None

def precision_score_(y, y_hat, pos_label=1):
	"""
		Compute the precision score.
	Args:
		y:a numpy.ndarray for the correct labels
		y_hat:a numpy.ndarray for the predicted labels
		pos_label: str or int, the class on which to report the precision_score (default=1)
	Return:
		The precision score as a float.
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	try:
		tp = 0
		fp = 0
		for i in range(0, len(y)):
			if y[i][0] == pos_label and y_hat[i][0] == pos_label:
				tp += 1
			if y_hat[i][0] == pos_label and y[i][0] != pos_label:
				fp += 1
		return tp / (tp + fp)
	except:
		return None

def recall_score_(y, y_hat, pos_label=1):
	"""
	Compute the recall score.
	Args:
		y:a numpy.ndarray for the correct labels
		y_hat:a numpy.ndarray for the predicted labels
		pos_label: str or int, the class on which to report the precision_score (default=1)
	Return:
		The recall score as a float.
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	try:
		tp = 0
		fn = 0
		for i in range(0, len(y)):
			if y[i][0] == pos_label and y_hat[i][0] == pos_label:
				tp += 1
			if y[i][0] == pos_label and y_hat[i][0] != pos_label:
				fn += 1
		return tp / (tp + fn)
	except:
		return None

def f1_score_(y, y_hat, pos_label=1):
	"""
	Compute the f1 score.
	Args:
		y:a numpy.ndarray for the correct labels
		y_hat:a numpy.ndarray for the predicted labels
		pos_label: str or int, the class on which to report the precision_score (default=1)
	Returns:
		The f1 score as a float.
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	try:
		pres = precision_score_(y, y_hat, pos_label)
		recall = recall_score_(y, y_hat, pos_label)
		return (2 * pres * recall) / (pres + recall)
	except:
		return None
