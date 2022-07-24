import numpy as np
from TinyStatistician import TinyStatistician as Ts

def zscore(x):
	if type(x).__module__ != np.__name__:
		return None
	if len(x) == 0 or x.shape[1] != 1:
		return None

	ts = Ts()

	mean = ts.mean(x)
	std = ts.std(x)
	x_prime = (x - mean) / std
	return x_prime
