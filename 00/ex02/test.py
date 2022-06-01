from prediction import simple_predict
import numpy as np

print(simple_predict(np.array([[2], [3], [4], [5]]), np.array([[2], [2]])))

x = np.arange(1,6).reshape(-1, 1)
# Example 1:
theta1 = np.array([[5],[0]])
print(simple_predict(x, theta1))
# Ouput:
print([[5.],[5.],[5.],[5.],[5.]])
# Do you understand why y_hat contains only 5’s here?
# Example 2:
theta2 = np.array([[0],[1]])
print(simple_predict(x, theta2))
# Output:
print([[1.],[2.],[3.],[4.],[5.]])
# Do you understand why y_hat == x here?
# Example 3:
theta3 = np.array([[5],[3]])
print(simple_predict(x, theta3))
# Output:
print([[ 8.],[11.],[14.],[17.],[20.]])
# Example 4:
theta4 = np.array([[-3],[1]])
print(simple_predict(x, theta4))
# Output:
print([[-2.],[-1.],[ 0.],[ 1.],[ 2.]])
