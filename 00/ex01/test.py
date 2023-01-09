import numpy as np
from TinyStatistician import TinyStatistician

def test_input(f):
	assert f(["salut"]) is None
	assert f([]) is None
	assert f([0, 3, "lol"]) is None
	assert f((1, 2, 3)) is None
	assert f(np.array(["salut"])) is None
	assert f(np.array([])) is None
	assert f(np.array([0, 3, "lol"])) is None
	assert f(np.array((1, 2, 3))) is not None
	assert f([1, 2, 3]) is not None
	assert f([1.54, 2.215, 3.177]) is not None
	assert f([1.54, 2.215, 3]) is not None
	assert f(np.array([1, 2, 3])) is not None
	assert f(np.array([1.54, 2.215, 3.177])) is not None
	assert f(np.array([1.54, 2.215, 3])) is not None

def test_mean():
	sta = TinyStatistician()
	test_input(sta.mean)
	assert sta.mean([564.654, 654460122.145, 64.25]) == np.array([564.654, 654460122.145, 64.25]).mean()
	assert sta.mean(np.array([564.654, 654460122.145, 64.25])) == np.array([564.654, 654460122.145, 64.25]).mean()
	assert sta.mean([1231.312]) == np.array([1231.312]).mean()
	assert sta.mean(np.array([1231.312])) == np.array([1231.312]).mean()
	assert sta.mean(np.array([0])) == np.array([0]).mean()


def test_median():
	sta = TinyStatistician()
	test_input(sta.median)
	pair = [1, 3, 4, 2]
	npair = np.array(pair)
	print("med")
	print(sta.median(pair))
	print(np.median(npair))
	# # assert sta.median(pair) == np.median(pair)
	# assert sta.median(npair) == np.median(np.array(pair))
	impair = [1, 3, 4, 2, 8, 9]
	nimpair = np.array(impair)
	print(sta.median(impair))
	print(np.median(nimpair))
	# assert sta.median(nimpair) == np.median(nimpair)
	# assert sta.median([253]) == np.median([253])
	# assert sta.median(np.array([253])) == np.median([253])
	print("Test 2 median")

	# print(sta.median(pair))

def test_quartile():
	sta = TinyStatistician()
	test_input(sta.quartile)
	print(int(1.75))
	print(sta.quartile([5, 4, 3, 2, 1]))
	print(np.quantile([5, 4, 3, 2, 1], 0.25))
	print(np.quantile([5, 4, 3, 2, 1], 0.75))
	print(sta.quartile([5, 4, 3, 2, 1, 6]))
	print(np.quantile([5, 4, 3, 2, 1, 6], 0.25))
	print(np.quantile([5, 4, 3, 2, 1, 6], 0.75))
	print(sta.quartile([1, 42, 300, 10, 59]))
	print(sta.quartile([1]))

def test_percentile():
	sta = TinyStatistician()
	ar = [1, 42, 300, 10, 59]
	# print(ar.sort())
	print(sta.percentile(ar, 10))
	print(sta.percentile(ar, 28))
	print(sta.percentile(ar, 83))
	print(sta.percentile(ar, 50))
	print(sta.median(ar))
	ar = [1, 42, 300, 10, 59, 411]
	# print(ar.sort())
	print(sta.percentile(ar, 10))
	print(sta.percentile(ar, 28))
	print(sta.percentile(ar, 83))
	print(sta.percentile(ar, 50))
	print(sta.median(ar))

def test_var():
	sta = TinyStatistician()
	ar = [1, 42, 300, 10, 59]
	print(sta.var(ar))

def test_std():
	sta = TinyStatistician()
	ar = [1, 42, 300, 10, 59]
	print(sta.std(ar))
# test_input()
test_mean()
test_median()
test_quartile()
test_percentile()
test_var()
test_std()

print("ici")
data = [42, 7, 69, 18, 352, 3, 650, 754, 438, 2659]
sta = TinyStatistician()
print("10: ", abs(sta.percentile(data, 10)))
