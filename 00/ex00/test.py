from matrix import Matrix
from matrix import Vector

m = Matrix([[1,2,3], [1, 2, 3]])
print(str(m))
m = Matrix([[1,2,3, 4], [1, 2, 3, 4]])
print(str(m))
m = Matrix([[1], [2]])
print(str(m))
m = Matrix([[1]])
print(str(m))
m = Matrix([[]])
print(str(m))
# m = Matrix([])
# print(str(m))

m = Matrix((1, 1))
print(str(m))
m = Matrix((2, 1))
print(str(m))
m = Matrix((5, 5))
print(str(m))
m = Matrix((4, 0))
print(str(m))

m = Matrix((3, 1))
print(str(m))

m = Matrix((1, 10))
print(str(m))

# try:
# m = Matrix([])
	# assert False
# except:
	# assert True
	# pass

# try:
# m = Matrix([1, 2, 3])
	# assert False
# except:
	# pass

# try:
	# assert False
# m = Matrix([[1, 2, 3], [1, 2]])
	# assert False
# except:
	# pass

m = Matrix([[1,2,3], [1, 2, 3]])
print(str(m))
m2 = Matrix([[1, 1, 1], [2, 2, 2]])
print(str(m2))
m3 = m + m2
print(str(m3))

m = Matrix([[1,2,3], [1, 2, 3]])
print(str(m))
m2 = Matrix([[1, 1, 1], [2, 2, 2]])
print(str(m2))
m3 = m - m2
print(str(m3))

m4 = m3 * 2

print(str(m4))
m4 = 2 * m3
print(str(m4))

m4 += m
print(str(m4))
m4 -= m
print(str(m4))
m4 - m
print(str(m4))

m5 = m / 2
print(str(m5))
m /= 2
print(str(m))
m /= 2
print(str(m))

print("--- VEctors ---")
v = Vector([[1], [2], [3]])
print(str(v))
v2 = Vector([[1, 2, 3]])
print(str(v2))
# v2 = Vector([[1, 2, 3], [1, 2, 3]])
# print(str(v2))
print(str(v + v))
print(type(v + v))
print(str(v - v))
print(type(v - v))
v3 = Vector([[2], [2], [2]])
v += v3
print(str(v))
print(type(v))
v -= v3
print(str(v))
print(type(v))
print(str(v * 2))
print(str(v / 2))
v *= 2
print(str(v))
v /= 2
print(str(v))
2 * v
print(str(2 * v))
2 / v
print(str(2 / v))
# print(1 - v)

m = Matrix([[1, 1, 1], [1, 1, 1]])
m2 = Matrix([[2, 2, 2], [2, 2, 2]])
print(str(m + m2))
print(str(m2 + m))
print(str(m - m2))
print(str(m2 - m))
print(str(m * 2))
print(str(2 * m))
print(str(m / 2))
print(str(2 / m))
# print(str(m + 2))
# print(str(2 + m))
m = Matrix([[10], [10], [10]])
print(str(m + v))
print(str(v + m))
# print(2 + v)
