import numpy as np
import pandas as pd
import sys
from my_linear_regression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt

def plot_res(x, y, y_pred):
	f = plt.figure("are_blue_pills_magics")
	f.clear()
	plt.plot(x, y_pred, color="green", linestyle="--", label="Spredict")
	plt.scatter(x, y, color="red", label="Strue")
	plt.scatter(x, y_pred, color="green")
	plt.legend()
	plt.xlabel("Quantity of blue pills (Micrograms)")
	plt.ylabel("Space driving score")
	#plt.plot(self.X, Y_pred, color="red")
	#plt.scatter(self.X, self.Y)
	plt.ioff()
	plt.show()

def plot_loss(x, y):
	t1 = np.linspace(-13, -4)
	print(t1.shape)
	# t0 = np.full((t1.shape[0], t1.shape[0]), t1)
	t0 = np.array([[87], [89], [91], [93]])
	print(t1, t0)
	f = plt.figure("are_blue_pills_magics")
	f.clear()
	# loss = []

	loss = []
	for j in range(0, len(t0)):
		loss = []
		for i in range(0, len(t1)):
			reg = MyLR([[t0[j]], [t1[i]]])
			loss.append(reg.loss_(y, reg.predict_(x)))
			# print(len(t1), len(loss))
		plt.plot(t1, loss, label=f"{t0[j]}")
	plt.legend()
	plt.ioff()
	plt.show()


reg = MyLR(np.array([[0], [0]]), alpha=0.001, max_iter=50000)
try:
	df = pd.read_csv("are_blue_pills_magics.csv")
except:
	print("Error while trying to open ./are_blue_pills_magics.csv")
	sys.exit(1)

x = df[['Micrograms']].to_numpy()
y = df[['Score']].to_numpy()
print(x)
print(y)

reg.fit_(x, y)
print(reg.thetas)
y_pred = reg.predict_(x)
print(reg.loss_elem_(y, y_pred))
print(reg.loss_(y, y_pred))

plot_loss(x, y)
plot_res(x, y, y_pred)
