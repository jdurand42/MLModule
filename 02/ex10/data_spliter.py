import numpy as np

def data_spliter(x, y, proportion):
	"""Shuffles and splits the dataset (given by x and y) into a training and a test set,
		while respecting the given proportion of examples to be kept in the training set.
	Args:
		x: has to be an numpy.array, a matrix of dimension m * n.
		y: has to be an numpy.array, a vector of dimension m * 1.
		proportion: has to be a float, the proportion of the dataset that will be assigned to the
		training set.
	Return:
		(x_train, x_test, y_train, y_test) as a tuple of numpy.array
		None if x or y is an empty numpy.array.
		None if x and y do not share compatible dimensions.
		None if x, y or proportion is not of expected type.
	Raises:
		This function should not raise any Exception.
	"""
	if type(x).__module__ != np.__name__:
		return None
	if type(y).__module__ != np.__name__:
		return None
	if len(x) == 0 or len(y) == 0:
		return None
	if x.shape[0] != y.shape[0] or y.shape[1] != 1:
		return None
	if isinstance(proportion, float) == False or proportion > 1 or proportion < 0:
		return None

	df = np.concatenate((x, y), axis=1)
	np.random.shuffle(df)
	x_i = [*range(0, df.shape[1] - 1)]
	x = df[:, x_i]
	y = df[:, [x.shape[1]]]

	p = int(len(x) * proportion)
	return (x[:p], x[p:], y[:p], y[p:])
