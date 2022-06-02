from matrix import Matrix
from matrix import Vector

def assert_test(f, good, error):
	try:
		f()
		print(good)
	except:
		# print(e)
		print(error)

def assert_error(f, excepted_e):
	try:
		f()
	except e:
		assert e == excepted_e

def test_init_matrix():
	m_1_3 = [[1.0, 2.0, 3.0]]
	m_2_3 = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]
	m = Matrix(m_1_3)
	m2 = Matrix(m_2_3)
	assert m_1_3 == m.data
	assert m_2_3 == m2.data
	assert m.shape == (1, 3)
	assert m2.shape == (2, 3)
	assert Matrix((1, 2)).shape == (1, 2)
	assert Matrix((1, 2)).data[0][0] == 0

	m3 = Matrix((10, 10))
	for i in range(0, len(m3.data)):
		for j in range(0, len(m3.data[i])):
			assert m3.data[i][j] == 0

	m = Matrix([[]])
	assert m.shape == (1, 0)
	assert m.data == [[]]

def test_init_vector():
	v = Vector([[1, 2, 3]])
	assert v.shape == (1, 3)
	assert v.data == [[1, 2, 3]]
	assert isinstance(v, Vector)
	v = Vector([[1], [2], [3]])
	assert v.data == [[1], [2], [3]]
	assert v.shape == (3, 1)
	assert isinstance(v, Vector)

	v = Vector((1, 3))
	# print(v.shape)
	# print(v.data)
	assert v.shape == (1, 3)
	assert v.data == [[0, 0, 0]]
	assert isinstance(v, Vector)

	v = Vector((3, 1))
	# print(v.shape)
	# print(v.data)
	assert v.shape == (3, 1)
	assert v.data == [[0], [0], [0]]
	assert isinstance(v, Vector)

def test_magic_matrixes():
	m = Matrix([[1, 1, 1]])
	m2 = Matrix([[3, 3, 3]])
	mb = m + m2
	assert mb.data == [[4, 4, 4]]
	assert mb.shape == (1, 3)
	assert m.data == [[1, 1, 1]]
	m = Matrix([[1, 1, 1], [1, 1, 1]])
	m2 = Matrix([[3, 3, 3], [3, 3, 3]])
	mb = m + m2
	assert mb.data == [[4, 4, 4], [4, 4, 4]]
	assert mb.shape == (2, 3)
	assert m.data == [[1, 1, 1], [1, 1, 1]]

def test_mul():
	m = Matrix([[0.0, 1.0, 2.0, 3.0],[0.0, 2.0, 4.0, 6.0]])
	m2 = Matrix([[0.0, 1.0],
	[2.0, 3.0],
	[4.0, 5.0],
	[6.0, 7.0]])
	print(m * m2)
	print(m2 * m)

	m1 = Matrix([[0.0, 1.0, 2.0],
	[0.0, 2.0, 4.0]])
	v1 = Vector([[1], [2], [3]])
	print(m1 * v1)
	print(v1 * m1)

	print(v1 * v1)

if __name__ == "__main__":
	print("--test init---")
	assert_test(test_init_matrix, "Init test are good", "Error in init")
	assert_test(test_init_vector, "Vector init test good", "Error in init test vector")
	assert_test(test_magic_matrixes, "Magic method on matrixes good", "Error in matrix magix")
	test_mul()
	# assert v.shape == (1, 3)
	# assert v.data == [[0, 0, 0]]
	# assert isinstance(v, Vector)
