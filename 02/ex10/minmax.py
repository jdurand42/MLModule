import numpy as np

def minmax(x):
	if type(x).__module__ != np.__name__:
		return None
	if len(x) == 0 or x.shape[1] != 1:
		return None

	x_prime = (x - x.min()) / (x.max() - x.min())
	return x_prime
