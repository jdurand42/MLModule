import numpy as np
from my_logistic_regression import MyLogisticRegression as MyLR
X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
Y = np.array([[1], [0], [1]])
thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
mylr = MyLR(thetas)
# Example 0:
print(mylr.predict_(X))
# Output:
# array([[0.99930437],
# [1. ],
# [1. ]])
# Example 1:
print(mylr.loss_(Y, mylr.predict_(X)))
# Output:
# 11.513157421577004
# Example 2:
print(mylr.fit_(X, Y))
print(mylr.thetas)
# Output:
# array([[ 2.11826435]
# [ 0.10154334]
# [ 6.43942899]
# [-5.10817488]
# [ 0.6212541 ]])
# Example 3:
print(mylr.predict_(X))
# Output:
# array([[0.57606717]
# [0.68599807]
# [0.06562156]])
# Example 4:
print("loss elem", mylr.loss_elem_(Y, mylr.predict_(X)))
print(mylr.loss_(Y, mylr.predict_(X)))
# Output:
# 1.477912692305226
