import numpy as np
import pandas as pd

def get_idx(x, label):
	for i in range(0, len(x)):
		if x[i] == label:
			return i
	return 0

def confusion_matrix_(y_true, y_hat, labels=None, df_option=False):
	"""
	Compute confusion matrix to evaluate the accuracy of a classification.
	Args:
		y:a numpy.array for the correct labels
		y_hat:a numpy.array for the predicted labels
		labels: optional, a list of labels to index the matrix.
		This may be used to reorder or select a subset of labels. (default=None)
		df_option: optional, if set to True the function will return a pandas DataFrame
		instead of a numpy array. (default=False)
	Return:
		The confusion matrix as a numpy array or a pandas DataFrame according to df_option value.
		None if any error.
	Raises:
		This function should not raise any Exception.
	"""
	uni = np.unique(np.concatenate((y_true, y_hat)))
	if labels is None:
		labels = uni
	dim = len(labels)
	mat = np.zeros((dim, dim), dtype=int)

	y = y_true
	for i in range(0, len(y)):
		if y[i][0] in labels:
			idx_label = get_idx(labels, y[i][0])
			if y[i][0] == y_hat[i][0]:
				mat[idx_label][idx_label] += 1
			elif y[i][0] != y_hat[i][0] and y_hat[i][0] in labels:
				mat[idx_label][get_idx(labels, y_hat[i][0])] += 1
	if df_option is False:
		return mat
	return pd.DataFrame(mat, columns=labels, index=labels)
	# Df options true
