import numpy as np
import pandas as pd
from my_logistic_regression import MyLogisticRegression as MyLR
from data_spliter import data_spliter
from z_score import zscore
import matplotlib.pyplot as plt
from mono_log import get_data, label_one_vs_all, get_numpy_feature, mono_log, binarise_y, get_absolute_score, plot_a_scatter, normalize_data

planet_labels = ["The flying cities of Venus", "United Nations of earth", "Mars Republic", "The asteroid's belt colonies"]

def label_one_vs_all_numpy(y, value):
	y = y.copy()
	for i in range(0, len(y)):
		if y[i][0] != value:
			y[i][0] = 0
		else:
			y[i][0] = 1
	# print(y.head())
	return y

def get_ys(Y):
	ys = []
	for i in range(0, 4):
		ys.append(label_one_vs_all_numpy(Y, i))
	return ys

def get_train_test(x, ys, treshold):
	sets = []

	for i in range(0, 4):
		sets.append(dict.fromkeys(['X_train', 'X_test', 'Y_train', 'Y_test']))
		# X_train, X_test, Y_train, Y_test = data_spliter(X.to_numpy(), ys[i].to_numpy(), treshold)
		sets[i]['X_train'], sets[i]['X_test'], sets[i]['Y_train'], sets[i]['Y_test'] \
		= data_spliter(X.to_numpy(), ys[i].to_numpy(), treshold)
	return sets

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
	X = normalize_data(X)
	X_train, X_test, Y_train, Y_test = data_spliter(X.to_numpy(), Y.to_numpy(), 0.7)

	ys_train = get_ys(Y_train)
	ys_test = get_ys(Y_test)
	# sets = get_train_test(X, ys, 0.7)

	regs = []
	y_preds = []
	for i in range(0, len(ys_train)):
		print(f"Performing a {planet_labels[i]} one vs all")
		regs.append(mono_log(X_train, ys_train[i]))
		y_pred = regs[i].predict_(X_test)
		y_preds.append(y_pred.copy())
		print(f"predict_ sample: {y_pred[:5]}")
		print(f"Loss: {regs[i].loss_(ys_test[i], y_pred)}")
		# print(f"Score: {regs[i].score_(ys_test[i], y_pred)}")
		print(f"Absolute score: {get_absolute_score(ys_test[i], binarise_y(y_pred, 0.5))}")

	print(f"Performing One vs all")
	final_y_pred = y_pred.copy()

	for i in range(0, len(y_pred)):
		y_pred[i][0] = y_preds[0][i][0]
		final_y_pred[i][0] = 0
		for j in range(0, len(y_preds)):
			# print(y_preds[j][i][0], y_pred[i][0], j)
			# print(y_preds[j][i][0])
			if y_preds[j][i][0] > y_pred[i][0]:
				final_y_pred[i][0] = j
		# print(final_y_pred[i][0])
	# print(final_y_pred)
	score = get_absolute_score(Y_test, final_y_pred)
	print(f"Score {score}")
	# y_pred_binary = binarise_y(y_pred, 0.5)

	# plot_a_scatter(X_test, Y_test, y_pred, 0, X.keys())
