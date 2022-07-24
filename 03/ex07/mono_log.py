import numpy as np
import pandas as pd
from my_logistic_regression import MyLogisticRegression as MyLR
from data_spliter import data_spliter
from z_score import zscore
import matplotlib.pyplot as plt

planet_labels = ["The flying cities of Venus", "United Nations of earth", "Mars Republic", "The asteroid's belt colonies"]

def get_data(path):
	try:
		df = pd.read_csv(path)
		return df
	except:
		raise LogicError(f"{path} data path not found")

def label_one_vs_all(y, target, value):
	y = y.copy()
	for i in range(0, len(y)):
		if y.loc[i, target] != value:
			y.loc[i, target] = 0
		elif y.loc[i, target] == value:
			y.loc[i, target] = 1
	# print(y.head())
	return y

def get_numpy_feature(df, key):
	return df[key].to_numpy().reshape((len(df[key]), 1))

def normalize_data(x):
	for key in x.keys():
		x[key] = zscore(get_numpy_feature(x, key))
	return x

def mono_log(x, y, thetas = None):
	if thetas is None:
		thetas = np.zeros((x.shape[1] + 1, 1))
	reg = MyLR(thetas, alpha=1e-6, max_iter=100000)
	# reg = MyLR(thetas)
	thetas = reg.fit_(x, y)
	print(f"Thetas: {thetas}")
	return reg

def binarise_y(y, treshold):
	y = y.copy()
	for i in range(0, len(y)):
		if y[i][0] >= treshold:
			y[i][0] = 1
		else:
			y[i][0] = 0
	return y

def get_absolute_score(y, y_pred):
	positives = 0
	for i in range(0, len(y)):
		if y[i][0] == y_pred[i][0]:
			positives += 1
	return positives / len(y)

def plot_a_scatter(x, y, y_pred, planet_code, features):
	plt.rcParams["figure.figsize"] = (15,7.5)
	f, ax = plt.subplots(3, 1)

	i = 0
	# x_sorted = np.sort(x, axis=0)
	# print(x_sorted)
	# y_pred = reg.predict_(x_sorted)
	for key in features:
		ax[i].scatter(x[:,[i]], y_pred, color="red", label="y_pred")
		ax[i].scatter(x[:,[i]], y, color="blue", label="y_true")
		ax[i].legend()
		ax[i].set_title(f"{key}")
		i += 1

	f.suptitle(f"{planet_labels[planet_code]} vs all")
	plt.ioff()
	plt.show()



if __name__=="__main__":
	X = get_data("./solar_system_census.csv")
	Y = get_data("./solar_system_census_planets.csv")
	X.drop(columns=X.columns[0],
        axis=1,
        inplace=True)
	Y.drop(columns=Y.columns[0],
        axis=1,
        inplace=True)
	print(X.head(5))
	print(Y.head(5))
	Y_venus = label_one_vs_all(Y, "Origin", 0)
	print(Y_venus.head(5))

	# X_true = X.copy()
	# X = normalize_data(X)
	# print(X.head())
	X_train, X_test, Y_train, Y_test = data_spliter(X.to_numpy(), Y_venus.to_numpy(), 0.7)
	print(f"Performing a {planet_labels[0]} one vs all")
	venus_reg = mono_log(X_train, Y_train)
	y_pred = venus_reg.predict_(X_test)
	print(f"predict_ sample: {y_pred[:5]}")
	print(f"Loss: {venus_reg.loss_(Y_test, y_pred)}")
	print(f"Score: {venus_reg.score_(Y_test, y_pred)}")

	y_pred_binary = binarise_y(y_pred, 0.5)
	print(f"Absolute score: {get_absolute_score(Y_test, y_pred_binary)}")

	plot_a_scatter(X_test, Y_test, y_pred, 0, X.keys())
