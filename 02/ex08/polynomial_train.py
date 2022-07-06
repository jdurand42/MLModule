import numpy as np
import pandas as pd
from mylinearregression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt
from polynomial_model import add_polynomial_features

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

def plot_sub_graph(regs, set, x, y):

	plt.rcParams["figure.figsize"] = (15,7.5)
	f, ax = plt.subplots(2, 3)

	true_x = x
	random_x = 6 * np.random.random_sample((100, 1)) + 1
	x = np.concatenate((x, random_x))
	x = np.sort(x, axis=0)
	# colors = ["indigo", "darkviolet", "thistle", "plum", "mediumblue", "slateblue"]
	for i in range(0, len(regs)):
		if i < 3:
			axis_x = 0
			axis_y = i
		else:
			axis_x = 1
			axis_y = i - 3
		ax1 = ax[axis_x, axis_y]

		ax1.scatter(true_x, y, color="blue", label="True data")

		x_to_pred = add_polynomial_features(x, i+1)
		y_pred = regs[i].predict_(x_to_pred)
		ax1.plot(x, y_pred, color="red", label=f"Predict with {i+1}'s degree")
		ax1.legend()
		# ax1.xlabel("Micrograms")
		# ax2.ylabel("Score")

	plt.ioff()
	plt.show()

def plot_big_graph(regs, set, x, y):

	plt.rcParams["figure.figsize"] = (15,7.5)
	f = plt.figure("Big graph")
	f.clear()

	plt.scatter(x, y, color="blue", label="True data")

	random_x = 6 * np.random.random_sample((100, 1)) + 1
	x = np.concatenate((x, random_x))
	x = np.sort(x, axis=0)

	colors = ["indigo", "green", "grey", "black", "pink", "red"]
	for i in range(0, len(regs)):
		x_to_pred = add_polynomial_features(x, i+1)
		y_pred = regs[i].predict_(x_to_pred)
		plt.plot(x, y_pred, color=colors[i], label=f"Predict with {i+1}'s degree")

	plt.xlabel("Micrograms")
	plt.ylabel("Score")
	plt.legend()
	plt.ioff()
	plt.show()


df = pd.read_csv("are_blue_pills_magics.csv")
print(df.head())

y = get_numpy_feature(df, "Score")
x = get_numpy_feature(df, "Micrograms")

set = []
regs = []
# theta1 =  np.array([[0.04965788], [0.18593411]])
theta1 = None
theta2 = None
theta3 = None
theta4 = np.array([[-20],[ 160],[ -80],[ 10],[ -1]]).reshape(-1,1)
theta5 = np.array([[1140],[ -1850],[ 1110],[ -305],[ 40],[ -2]]).reshape(-1,1)
theta6 = np.array([[9110],[ -18015],[ 13400],[ -4935],[ 966],[ -96.4],[ 3.86]]).reshape(-1,1)
thetas = [theta1, theta2, theta3, theta4, theta5, theta6]

alphas = [0.01, 0.001, 0.00001, 0.000001, 0.00000001, 0.000000001]
# 3 must be 1000000
max_iters = [100000, 100000, 1000000, 100000, 100000, 100000]

for i in range(0, 6):
	set.append(add_polynomial_features(x, i+1))
	print(f"Training {i+1} power")
	regs.append(get_multivariate_model(set[i], y, thetas=thetas[i], alpha=alphas[i], max_iter=max_iters[i]))
	compute(regs[i], set[i], y, f"Micrograms power {i+1}", "Score")
	print("")
	# plot_univariate(x, y, regs[i].predict_(set[i]), "bla", "bloe")

plot_polymonial_error(regs, y, set)

plot_sub_graph(regs, set, x, y)
plot_big_graph(regs, set, x, y)
