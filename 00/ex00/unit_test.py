import unittest
from matrix import Matrix, Vector
import numpy as np

class TestMatrixInit(unittest.TestCase):
    def test_matrix_init(self):
        m_1_3 = [[1.0, 2.0, 3.0]]
        m_2_3 = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]
        m = Matrix(m_1_3)
        m2 = Matrix(m_2_3)
        self.assertEqual(m_1_3, m.data)
        self.assertEqual(m_2_3, m2.data)
        self.assertEqual(m.shape, (1, 3))
        self.assertEqual(m2.shape, (2, 3))
        self.assertEqual(Matrix((1, 2)).shape, (1, 2))
        self.assertEqual(Matrix((1, 2)).data[0][0], 0)

        m3 = Matrix((10, 10))
        for i in range(0, len(m3.data)):
            for j in range(0, len(m3.data[i])):
                self.assertEqual(m3.data[i][j], 0)

        m = Matrix([[]])
        self.assertEqual(m.shape, (1, 0))
        self.assertEqual(m.data, [[]])
    
        with self.assertRaises(TypeError):
            m = Matrix([1, 2])
        with self.assertRaises(TypeError):
            m = Matrix((-1, 12))
            print(m.data)
        with self.assertRaises(TypeError):
            m = Matrix(5)
        with self.assertRaises(TypeError):
            m = Matrix()
        with self.assertRaises(TypeError):
            m = Matrix([[[1, 2]]])
    
    def test_repr_str(self):
        m =Matrix((1 , 1))
        self.assertTrue(len(str(m))> 0)
        self.assertTrue(isinstance(str(m), str))
        self.assertTrue(len(repr(m))> 0)
        self.assertTrue(isinstance(repr(m), str))
        m =Vector((1 , 1))
        self.assertTrue(len(str(m))> 0)
        self.assertTrue(isinstance(str(m), str))
        self.assertTrue(len(repr(m))> 0)
        self.assertTrue(isinstance(repr(m), str))

    def test_vector_init(self):
        v = Vector([[1, 2, 3]])
        self.assertEqual(v.shape, (1, 3))
        self.assertEqual(v.data, [[1, 2, 3]])
        self.assertTrue(isinstance(v, Vector))
        v = Vector([[1], [2], [3]])
        self.assertEqual(v.data, [[1], [2], [3]])
        self.assertEqual(v.shape, (3, 1))
        self.assertTrue(isinstance(v, Vector))

        v = Vector((1, 3))
        # print(v.shape)
        # print(v.data)
        self.assertEqual(v.shape, (1, 3))
        self.assertEqual(v.data, [[0, 0, 0]])
        self.assertTrue(isinstance(v, Vector))

        v = Vector((3, 1))
        # print(v.shape)
        # print(v.data)
        self.assertEqual(v.shape, (3, 1))
        self.assertEqual(v.data, [[0], [0], [0]])
        self.assertTrue(isinstance(v, Vector))

        with self.assertRaises(TypeError):
            v = Vector((-1, 12))
        with self.assertRaises(TypeError):
            v = Vector([1, 2])
        with self.assertRaises(TypeError):
            v = Vector([[1, 2], [1, 2]])
        with self.assertRaises(TypeError):
            v = Vector(5)
        with self.assertRaises(TypeError):
            v = Vector()

class TestMatrixOp(unittest.TestCase):
    def test_add_radd(self):
        m = Matrix([[1, 1, 1]])
        m2 = Matrix([[3, 3, 3]])
        mb = m + m2
        self.assertEqual(mb.data, [[4, 4, 4]])
        self.assertEqual(mb.shape, (1, 3))
        self.assertEqual(m.data, [[1, 1, 1]])
        self.assertEqual(m2.data, [[3, 3, 3]])
        mb = m2 + m
        self.assertEqual(mb.data, [[4, 4, 4]])
        self.assertEqual(mb.shape, (1, 3))
        self.assertEqual(m.data, [[1, 1, 1]])
        self.assertEqual(m2.data, [[3, 3, 3]])

        self.assertTrue(isinstance(mb, Matrix))

        m = Matrix([[1, 1, 1], [1, 1, 1]])
        m2 = Matrix([[3, 3, 3], [3, 3, 3]])
        mb = m + m2
        self.assertEqual(mb.data, [[4, 4, 4], [4, 4, 4]])
        self.assertEqual(mb.shape, (2, 3))
        self.assertEqual(m.data, [[1, 1, 1], [1, 1, 1]])
        mb = m2 + m
        self.assertEqual(mb.data, [[4, 4, 4], [4, 4, 4]])
        self.assertEqual(mb.shape, (2, 3))
        self.assertEqual(m.data, [[1, 1, 1], [1, 1, 1]])
        self.assertTrue(isinstance(mb, Matrix))

        m = Matrix((1, 3))
        m2 = Matrix((2, 3))
        with self.assertRaises(TypeError):
            mb = 3 + m
        with self.assertRaises(TypeError):
            mb = m + m2
        with self.assertRaises(TypeError):
            mb = m + 3

    def test_sub_rsub(self):
        m = Matrix([[1, 1, 1]])
        m2 = Matrix([[3, 3, 3]])
        mb = m - m2
        self.assertEqual(mb.data,[[-2, -2, -2]])
        self.assertEqual(mb.shape, (1, 3))
        self.assertEqual(m.data, [[1, 1, 1]])
        self.assertEqual(m2.data, [[3, 3, 3]])
        self.assertTrue(isinstance(mb, Matrix))
        mb = m2 - m
        self.assertEqual(mb.data,[[2, 2, 2]])
        self.assertEqual(mb.shape, (1, 3))
        self.assertEqual(m.data, [[1, 1, 1]])
        self.assertEqual(m2.data, [[3, 3, 3]])
        self.assertTrue(isinstance(mb, Matrix))

        m = Matrix([[1, 1, 1], [1, 1, 1]])
        m2 = Matrix([[3, 3, 3], [3, 3, 3]])
        mb = m - m2
        self.assertEqual(mb.data,[[-2, -2, -2],[-2, -2, -2]])
        self.assertEqual(mb.shape, (2, 3))
        self.assertEqual(m.data, [[1, 1, 1], [1, 1, 1]])
        self.assertEqual(m2.data, [[3, 3, 3], [3, 3, 3]])
        self.assertTrue(isinstance(mb, Matrix))
        mb = m2 - m
        self.assertEqual(mb.data,[[2, 2, 2], [2, 2, 2]])
        self.assertEqual(mb.shape, (2, 3))
        self.assertEqual(m.data, [[1, 1, 1], [1, 1, 1]])
        self.assertEqual(m2.data, [[3, 3, 3], [3, 3, 3]])
        self.assertTrue(isinstance(mb, Matrix))

        m = Matrix((1, 3))
        m2 = Matrix((2, 3))
        with self.assertRaises(TypeError):
            mb = 3 - m
        with self.assertRaises(TypeError):
            mb = m - 3
        with self.assertRaises(TypeError):
            mb = m - m2

    def test_mul_rmul(self):
        # scalar
        m = Matrix([[2, 3]])
        mb = m * 2
        self.assertEqual(mb.data, [[4, 6]])
        self.assertEqual(m.data, [[2, 3]])
        self.assertTrue(isinstance(mb, Matrix))
        mb = 2 * m
        self.assertEqual(mb.data, [[4, 6]])
        self.assertEqual(m.data, [[2, 3]])
        self.assertTrue(isinstance(mb, Matrix))

        # mb = m * m
        m = Matrix([[0.0, 1.0, 2.0, 3.0],[0.0, 2.0, 4.0, 6.0]])
        m2 = Matrix([[0.0, 1.0],
        [2.0, 3.0],
        [4.0, 5.0],
        [6.0, 7.0]])
        mb = m * m2
        # print(mb.data)
        self.assertEqual(mb.data, [[28.0, 34.0], [56.0, 68.0]])
        self.assertTrue(isinstance(mb, Matrix))
        mb = m2 * m
        # print(mb.data)
        self.assertEqual(mb.data, [[0.0, 2.0, 4.0, 6.0], [0.0, 8.0, 16.0, 24.0], [0.0, 14.0, 28.0, 42.0], [0.0, 20.0, 40.0, 60.0]])
        self.assertTrue(isinstance(mb, Matrix))

        m = Matrix([[2, 2, 2], [2, 2, 2]])
        m2 = Matrix([[3], [3], [3]])
        mb = m * m2
        self.assertTrue(mb.data, [[18], [18]])
        with self.assertRaises(TypeError):
            mb = m2 * m
        with self.assertRaises(TypeError):
            mb = m2 * m2

    def test_truediv_rtruediv(self):
        m = Matrix([[2, 4, 8], [2, 4, 8]])
        mb = m / 2
        self.assertEqual(mb.data, [[1, 2, 4], [1, 2, 4]])
        self.assertTrue(isinstance(mb, Matrix))
        mb = 2 / m
        self.assertTrue(isinstance(mb, Matrix))
        self.assertEqual(mb.data, [[1.0, 0.5, 0.25], [1.0, 0.5, 0.25]])

        with self.assertRaises(TypeError):
            m / m
        with self.assertRaises(TypeError):
            m / "salut"
        with self.assertRaises(TypeError):
            "saut" / m

    def test_t(self):
        m = Matrix((3 , 2))
        self.assertEqual(m.T().shape, (2, 3))
        self.assertEqual(m.shape, (3, 2))
        self.assertEqual(m.T().T().shape, (3, 2))
        m = Matrix((10,10))
        self.assertEqual(m.T().shape, (10, 10))

class TestVectorOp(unittest.TestCase):
    def test_add_radd(self):
        v = Vector([[1, 2, 3]])
        v2 = Vector([[1], [2], [3]])
        vb = v + v
        self.assertEqual(vb.data, [[2, 4, 6]])
        self.assertEqual(vb.shape, v.shape)
        self.assertTrue(isinstance(vb, Vector))
        vb = v2 + v2
        self.assertEqual(vb.data, [[2], [4], [6]])
        self.assertEqual(vb.shape, v2.shape)
        self.assertTrue(isinstance(vb, Vector))
        
        m = Matrix([[1, 2, 3]])
        vb = v + m
        self.assertEqual(vb.data, [[2, 4, 6]])
        self.assertEqual(vb.shape, v.shape)
        self.assertTrue(isinstance(vb, Vector))

        vb = m + v
        self.assertEqual(vb.data, [[2, 4, 6]])
        self.assertEqual(vb.shape, v.shape)
        self.assertTrue(isinstance(vb, Vector))


        with self.assertRaises(TypeError):
            mb = 3 + v
        with self.assertRaises(TypeError):
            mb = v + v2
        with self.assertRaises(TypeError):
            mb = v + 3
        with self.assertRaises(TypeError):
            mb = "3" + v

    def test_sub_rsub(self):
        m = Vector([[1, 1, 1]])
        m2 = Vector([[3, 3, 3]])
        mb = m - m2
        self.assertEqual(mb.data,[[-2, -2, -2]])
        self.assertEqual(mb.shape, (1, 3))
        self.assertEqual(m.data, [[1, 1, 1]])
        self.assertEqual(m2.data, [[3, 3, 3]])
        self.assertTrue(isinstance(mb, Vector))
        mb = m2 - m
        self.assertEqual(mb.data,[[2, 2, 2]])
        self.assertEqual(mb.shape, (1, 3))
        self.assertEqual(m.data, [[1, 1, 1]])
        self.assertEqual(m2.data, [[3, 3, 3]])
        self.assertTrue(isinstance(mb, Vector))

        m = Matrix([[1, 2, 3]])
        v = Vector([[2, 4, 6]])
        vb = v - m
        self.assertEqual(vb.data, [[1, 2, 3]])
        self.assertEqual(vb.shape, v.shape)
        self.assertTrue(isinstance(vb, Vector))

        vb = m - v
        self.assertEqual(vb.data, [[-1, -2, -3]])
        self.assertEqual(vb.shape, v.shape)
        self.assertTrue(isinstance(vb, Vector))

        m = Vector((1, 3))
        m2 = Vector((2, 1))
        with self.assertRaises(TypeError):
            mb = 3 - m
        with self.assertRaises(TypeError):
            mb = m - 3
        with self.assertRaises(TypeError):
            mb = m - m2
    

    def test_truediv_rtruediv(self):
        m = Vector([[2], [4]])
        mb = m / 2
        self.assertEqual(mb.data, [[1], [2]])
        self.assertTrue(isinstance(mb, Vector))
        mb = 2 / m
        self.assertTrue(isinstance(mb, Vector))
        self.assertEqual(mb.data, [[1], [0.5]])

        with self.assertRaises(TypeError):
            m / m
        with self.assertRaises(TypeError):
            m / "salut"
        with self.assertRaises(TypeError):
            "saut" / m
    
    def test_m_v_mul(self):
        m = Vector([[2, 3]])
        mb = m * 2
        self.assertEqual(mb.data, [[4, 6]])
        self.assertEqual(m.data, [[2, 3]])
        self.assertTrue(isinstance(mb, Vector))
        mb = 2 * m
        self.assertEqual(mb.data, [[4, 6]])
        self.assertEqual(m.data, [[2, 3]])
        self.assertTrue(isinstance(mb, Vector))

        # mb = m * m
        m = Vector([[0.0, 1.0, 2.0, 3.0]])
        m2 = Vector([[0.0],
        [2.0],
        [4.0],
        [6.0]])

        m1 = Matrix([[1, 2, 3]])
        v1 = Vector([[1], [2], [3]])
        mb = m1 * v1
        # print(mb.data)
        # print((Matrix(m1.data) * Matrix(v1.data)).data)
        self.assertEqual(mb.data, (Matrix(m1.data) * Matrix(v1.data)).data)
        self.assertTrue(isinstance(mb, Vector))
        mb = v1 * m1
        # print(mb.data, "ici")
        self.assertEqual(mb.data, (Matrix(v1.data) * Matrix(m1.data)).data)
        self.assertTrue(isinstance(mb, Matrix))

        m = Vector([[2], [2]])
        m2 = Vector([[3], [3], [3]])
        # mb = m * m2
        # self.assertTrue(mb.data, [[18], [18]])
        with self.assertRaises(TypeError):
            mb = m2 * m
        with self.assertRaises(TypeError):
            mb = m2 * m2
        with self.assertRaises(TypeError):
            mb = m2 * [[1, 2, 3]]
        with self.assertRaises(TypeError):
            mb = m2 * "dsds"
        with self.assertRaises(TypeError):
            mb = "dsdas" * m2
    
    def test_v_v_mul(self):
        m1 = Vector([[1, 2, 3]])
        m2 = Vector([[1], [2], [3]])

        mb = m1 * m2
        self.assertEqual(mb.data, (Matrix(m1.data) * Matrix(m2.data)).data)
        self.assertTrue(isinstance(mb, Vector))

        mb = m2 * m1
        self.assertEqual(mb.data, (Matrix(m2.data) * Matrix(m1.data)).data)
        self.assertTrue(isinstance(mb, Matrix))
    
    def test_t(self):
        m = Vector((3 , 1))
        self.assertEqual(m.T().shape, (1, 3))
        self.assertEqual(m.shape, (3, 1))
        self.assertEqual(m.T().T().shape, (3, 1))
        self.assertTrue(isinstance(m.T(), Vector))
        m = Vector((1 , 3))
        self.assertEqual(m.T().shape, (3, 1))
        self.assertEqual(m.shape, (1, 3))
        self.assertEqual(m.T().T().shape, (1, 3))
        self.assertTrue(isinstance(m.T(), Vector))

    def test_dot(self):
        v = Vector([[1], [2], [3]])
        v2 = Vector([[1], [2], [3]])

        mb = v.dot(v2)
        self.assertEqual(mb, 14)
        mb = v2.dot(v)
        self.assertEqual(mb, 14)

        v = Vector([[1 , 2, 3]])
        v2 = Vector([[1 , 2, 3]])
        mb = v.dot(v2)
        self.assertEqual(mb, 14)
        mb = v2.dot(v)
        self.assertEqual(mb, 14)

        with self.assertRaises(TypeError):
            v.dot(13)
        with self.assertRaises(TypeError):
            v.dot("lol")
        with self.assertRaises(TypeError):
            v.dot([[25, 1, 3]])

if __name__=="__main__":
    unittest.main()