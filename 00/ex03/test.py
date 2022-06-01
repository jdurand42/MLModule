import numpy as np
from tools import add_intercept
# Example 1:
x = np.arange(1,6).reshape((5,1))
print(x)
print(add_intercept(x))
print(x)
# Output:
# array([[1., 1.],
# [1., 2.],
# [1., 3.],
# [1., 4.],
# [1., 5.]])
# Example 2:
y = np.arange(1,10).reshape((3,3))
print(y)
print(add_intercept(y))
print(y)
# Output:
# array([[1., 1., 2., 3.],
# [1., 4., 5., 6.],
# [1., 7., 8., 9.]])
y = np.array([])
print(y)
print(add_intercept(y))
# Not sure
y = np.array([[]])
print(y)
print(add_intercept(y))
