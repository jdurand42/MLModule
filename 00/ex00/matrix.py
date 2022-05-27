
class Matrix():
	# self.data = []
	# self.shape = ()
	def __init__(self, arg):
		self.data = []
		try:
			if isinstance(arg, list):
				# No params check like len of matrix to be exact
				# Check for only int and float ?
				self.data = arg
				self.shape = (len(arg), len(arg[0]))
				for el in self.data:
					if len(el) != self.shape[1]:
						raise Exception("Error: Shape must be correct")
			elif isinstance(arg, tuple):
				self.shape = arg
				if len(arg) != 2:
					raise Exception("Error: Shape must be correct: (m, n)")
				for i in range(0, arg[0]):
					b = []
					for j in range(0, arg[1]):
						b.append(0)
					self.data.append(b)
		except:
			raise TypeError("Error: incorrect parameters for Matrix")

	def __str__(self):
		return (f"{type(self)}\n{self.shape}\n{self.data}")

	def __add__(self, other):
		if isinstance(other, Matrix) == False:
			raise TypeError("Error: Invalid opperand")
		if other.shape != self.shape:
			raise TypeError("Error: Can only add matrix of the same dimensions")

		b = []
		for i in range(0, len(self.data)):
			b.append([])
			for j in range(0, len(self.data[i])):
				b[i].append(self.data[i][j] + other.data[i][j])
		return Matrix(b)

	def __radd__(self, other):
		# Not implemented if object is not a Matrix or vector, doesn't throw right error
		return self.__add__(other)

	def __sub__(self, other):
		if isinstance(other, Matrix) == False:
			raise TypeError("Error: Invalid opperand")
		if other.shape != self.shape:
			raise TypeError("Error: Can only add matrix of the same dimensions")

		b = []
		for i in range(0, len(self.data)):
			b.append([])
			for j in range(0, len(self.data[i])):
				b[i].append(self.data[i][j] - other.data[i][j])
		return Matrix(b)

	def __rsub__(self, other):
		return self.__sub__(other)

	def __opscalar__(self, n, mult=True):
		b = []
		for i in range(0, len(self.data)):
			b.append([])
			for j in range(0, len(self.data[i])):
				if mult == True:
					b[i].append(self.data[i][j] * n)
				else:
					b[i].append(self.data[i][j] / n)
		return Matrix(b)

	def __mul__(self, other):
		if isinstance(other, (float, int)):
			return self.__opscalar__(other)

	def __rmul__(self, other):
		return self.__mul__(other)

	def __truediv__(self, other):
		if isinstance(other, (float, int)) == False:
			raise TypeError("Invalid opperand: Matrix div only works with scalars")
		return self.__opscalar__(other, mult=False)

	def __rtruediv__(self, other):
		if isinstance(other, (float, int)) == False:
			raise TypeError("Invalid opperand: Matrix div only works with scalars")
		b = []
		for i in range(0, len(self.data)):
			b.append([])
			for j in range(0, len(self.data[i])):
				b[i].append(other / self.data[i][j])
		return Matrix(b)

class Vector(Matrix):
	def __init__(self, arg):
		try:
			if isinstance(arg, list):
				if len(arg) > 1:
					for el in arg:
						if len(el) > 1:
							raise TypeError("Error: vector must be of shpe (1 * n) or (n * 1)")
			elif isinstance(arg, tuple):
				if len(arg) != 2 or 1 not in arg:
					raise TypeError("Error: vector must be of shpe (1 * n) or (n * 1)")
			super().__init__(arg)
		except:
			raise TypeError("Error: Vector arg must be of the form [[]] and of shape (1 * n) or (n * 1)")

	def __add__(self, other):
		# peut check ici si vector
		return Vector(super().__add__(other).data)

	# def __radd__(self, other):
	# 	# peut check ici si vector
	# 	return Vector(super().__radd__(other).data)

	def __sub__(self, other):
		return Vector(super().__sub__(other).data)

	# def __rsub__(self, other):
	# 	return self.__rsub__(other)

	def __mul__(self, other):
		return Vector(super().__mul__(other).data)

	def __rmul__(self, other):
		return Vector(super().__rmul__(other).data)

	def __truediv__(self, other):
		return Vector(super().__truediv__(other).data)

	def __rtruediv__(self, other):
		return Vector(super().__rtruediv__(other).data)
