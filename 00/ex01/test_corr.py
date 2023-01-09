import TinyStatistician as ts
import unittest
import numpy as np

data = [42, 7, 69, 18, 352, 3, 650, 754, 438, 2659]
epsilon = 1e-5
err = "Error, grade 0 :("


tstat = ts.TinyStatistician()
# assert abs(tstat.mean(data) - 499.2) < epsilon, err
# assert abs(tstat.median(data) - 210.5) < epsilon, err

# quartile = tstat.quartile(data)
# assert abs(quartile[0] - 18) < epsilon, err
# assert abs(quartile[1] - 650) < epsilon, err

# assert abs(tstat.percentile(data, 10) - 3) < epsilon, err
# assert abs(tstat.percentile(data, 28) - 18) < epsilon, err
# assert abs(tstat.percentile(data, 83) - 754) < epsilon, err

# print(tstat.var(data) - 654661)
# assert abs(tstat.var(data) - 654661) < epsilon, err
# print(tstat.std(data))
# assert abs(tstat.std(data) - 809.11) < epsilon, err

class TestTs(unittest.TestCase):
    def test_mean(self):
        # self.assertTrue(abs(tstat.mean(data) - 499.2) < epsilon)
        pass

    def test_median(self):
        self.assertTrue(abs(tstat.median(data) - 210.5) < epsilon)
    
    def test_quartile(self):
        quartile = tstat.quartile(data)
        # self.assertTrue(abs(quartile[0] - 18) < epsilon)
        # self.assertTrue(abs(quartile[1] - 650) < epsilon)
    
    def test_percentile(self):
        #  problem: 10 * 0.1 == 1
        # ils veulent le premier element donc 0
        print("numpy: ")
        print(np.percentile(np.array(data), 10))
        print("perc: ", abs(tstat.percentile(data, 10) - 3))
        self.assertTrue(abs(tstat.percentile(data, 10) - 3) < epsilon)
        # self.assertTrue(abs(tstat.percentile(data, 28) - 18) < epsilon)
        # self.assertTrue(abs(tstat.percentile(data, 83) - 754) < epsilon)
    
    def test_var(self):
        print("var: ", abs(tstat.var(data)))
        print("numpy: ", abs(np.var(np.array(data))))
        self.assertTrue(abs(tstat.var(data) - 654661) < epsilon)
    
    def test_std(self):
        print("std: ", abs(tstat.std(data)))
        self.assertTrue(abs(tstat.std(data) - 809.11) < epsilon)

if __name__ == "__main__":
    unittest.main()