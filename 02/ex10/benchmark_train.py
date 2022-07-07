import numpy as np
import pandas as pd
from mylinearregression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt
from polynomial_model import add_polynomial_features
from data_spliter import data_spliter
from minmax import minmax

def get_numpy_feature(df, key):
	return df[key].to_numpy().reshape((len(df[key]), 1))

def get_numpy_multi_features(df, keys):
	nkeys = len(keys)
	return df[keys].to_numpy().reshape((len(df[keys]), nkeys))

def get_univariate_model(x, y, thetas=None, **kwargs):
	if thetas is None:
		thetas = np.zeros((2, 1))
	reg = MyLR(thetas, **kwargs)
	reg.fit_(x, y)
	return reg

def get_multivariate_model(x, y, thetas=None, **kwargs):
	if thetas is None:
		thetas = np.zeros((x.shape[1] + 1, 1))
	reg = MyLR(thetas, **kwargs)
	reg.fit_(x, y)
	return reg

def compute(reg, x, y, feature, target):
	# print(reg.predict_(x))
	print(f"{feature} thetas: ", reg.thetas)
	y_pred = reg.predict_(x)
	print(f"{feature} loss:", reg.loss_(y, y_pred))
	print(f"{feature} R2:", reg.score(y, y_pred))
	return y_pred

def plot_univariate(x, y, y_pred, xlabel, ylabel):
	f = plt.figure(f"spacecraft_data: {ylabel} by {xlabel}")
	f.clear()
	plt.scatter(x, y_pred, color="green", linestyle="--", label="Spredict")
	plt.scatter(x, y, color="red", label="Strue")
	# plt.scatter(x, y_pred, color="green")
	plt.legend()
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	#plt.plot(self.X, Y_pred, color="red")
	#plt.scatter(self.X, self.Y)
	plt.ioff()
	plt.show()

def plot_polymonial_error(regs, y, set):
	f = plt.figure("Error by power")
	f.clear()
	for i in range(0, len(regs)):
		y_pred = regs[i].predict_(set[i])
		loss = regs[i].loss_(y, y_pred)
		plt.plot([i + 1, i+1], [0, loss], color="red")
	plt.xlabel("polynomial degree")
	plt.ylabel("MSE")
	plt.ioff()
	plt.show()

def plot_sub_graph(regs, set, x, y, feature=0):

	plt.rcParams["figure.figsize"] = (15,7.5)
	f, ax = plt.subplots(2, 2)

	true_x = x
	# random_x = 6 * np.random.random_sample((100, 1)) + 1
	# x = np.concatenate((x, random_x))
	# x = np.sort(x, axis=0)
	# colors = ["indigo", "darkviolet", "thistle", "plum", "mediumblue", "slateblue"]
	for i in range(0, len(regs)):
		if i < 2:
			axis_x = 0
			axis_y = i
		else:
			axis_x = 1
			axis_y = i - 2
		ax1 = ax[axis_x, axis_y]

		ax1.scatter(x[:,[feature]], y, color="blue", label="True data")

		# x_to_pred = add_polynomial_features(x, i+1)
		y_pred = regs[i].predict_(set[i])
		ax1.scatter(x[:,[feature]], y_pred, color="red", label=f"Predict with {i+1}'s degree")
		ax1.legend()
		# ax1.xlabel("Micrograms")
		# ax2.ylabel("Score")

	plt.ioff()
	plt.show()

df = pd.read_csv("space_avocado.csv")
print(df.head())
# df.head(10).to_csv("space_avocado_light.csv")
features = ["weight", "prod_distance", "time_delivery"]
# xy = get_numpy_multi_features(df, [features]+["target"])

y_true = get_numpy_feature(df, "target")
x_true = get_numpy_multi_features(df, features)

for key in features:
	# print(df[key].to_numpy().r)
	df[key] = minmax(get_numpy_feature(df, key))
print(df.head())
y = get_numpy_feature(df, "target")
x = get_numpy_multi_features(df, features)


print("la")

x_train, x_test, y_train, y_test = data_spliter(x, y, 0.5)

train_set = []
test_set = []
regs = []

theta1 = None
theta2 = None
theta3 = None
theta4 = None
thetas = [theta1, theta2, theta3, theta4]

alphas = [1e-2, 1e-2, 1e-2, 1e-2]
# 3 must be 1000000
max_iters = [10000, 10000, 10000, 10000]

print("")
print("X_train: ", x_train[:2])
print("")
print(f"train size: {x_train.shape}, test size: {y_test.shape}")

for i in range(0, 4):
	print(f"Training {i+1} power")
	subset = []
	for j in range(0, x_train.shape[1]):
		# print(x_train[:,[j]])
		poly = add_polynomial_features(x_train[:,[j]], i+1)
		# print(add_polynomial_features(x_train[:,[j]], i+1))
		subset.append(poly)
		# subset = np.concatenate(subset, add_polynomial_features(x_train[:,[j]], i+1))
	b = subset[0]
	for k in range(1, len(subset)):
		b = np.concatenate((b, subset[k]), axis=1)

	train_set.append(b)

	subset = []
	for j in range(0, x_test.shape[1]):
		# print(x_test[:,[j]])
		poly = add_polynomial_features(x_test[:,[j]], i+1)
		# print(add_polynomial_features(x_train[:,[j]], i+1))
		subset.append(poly)
		# subset = np.concatenate(subset, add_polynomial_features(x_train[:,[j]], i+1))
	b = subset[0]
	for k in range(1, len(subset)):
		b = np.concatenate((b, subset[k]), axis=1)
	test_set.append(b)
	# print(f"power: {i+1}, {b}")
	# print(b)


	# print("set: ", set[i])
	regs.append(get_multivariate_model(train_set[i], y_train, thetas=thetas[i], alpha=alphas[i], max_iter=max_iters[i]))
	compute(regs[i], test_set[i], y_test, f"Power {i+1}", "target")
	# compute(regs[i], train_set[i], y_train, f"Power {i+1}", "target")
	print("")
	# plot_univariate(x, y, regs[i].predict_(set[i]), "bla", "bloe")

# plot_polymonial_error(regs, y, set)

plot_sub_graph(regs, test_set, x_test, y_test)
plot_sub_graph(regs, test_set, x_test, y_test, feature=1)
plot_sub_graph(regs, test_set, x_test, y_test, feature=2)
# plot_big_graph(regs, set, x, y)
