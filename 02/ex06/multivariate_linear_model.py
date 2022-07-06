import numpy as np
import pandas as pd
from mylinearregression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt

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

df = pd.read_csv("spacecraft_data.csv")

y = get_numpy_feature(df, "Sell_price")

x = get_numpy_feature(df, "Age")
# [[634.4506121 ]
#  [-11.94720736]]
reg_age = get_univariate_model(x, y, alpha=0.001, max_iter=42000)
compute(reg_age, x, y, "Age", "Sell_price")
plot_univariate(x, y, reg_age.predict_(x), "Age", "Sell_price")

x = get_numpy_feature(df, "Thrust_power")
reg_thrust = get_univariate_model(x, y, alpha=0.0001)
compute(reg_thrust, x, y, "Thrust_power", "Sell_price")
plot_univariate(x, y, reg_thrust.predict_(x), "Thrust_power", "Sell_price")

x = get_numpy_feature(df, "Terameters")
reg_tera = get_univariate_model(x, y, alpha=0.0001, max_iter=420000)
compute(reg_tera, x, y, "Terameters", "Sell_price")
plot_univariate(x, y, reg_tera.predict_(x), "Terameters", "Sell_price")

# Multivariate:
features = ["Age", "Thrust_power", "Terameters"]
x = get_numpy_multi_features(df, features)
multi = get_multivariate_model(x, y, alpha=0.00001)
y_pred = compute(multi, x, y, features, "Sell price")

for i in range(0, len(features)):
	plot_univariate(get_numpy_feature(df, features[i]), y, y_pred, features[i], "Sell price")
# print(multi.thetas)
# y_pred = multi.predict_(x)
# print(multi.loss_(y, y_pred))
# print(multi.score(y, y_pred))
