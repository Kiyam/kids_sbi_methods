import unittest
import kcap_methods as km
import numpy as np

class TestKCAPMethods(unittest.TestCase):
    def test_get_params(self):
        vals = km.get_params(mock_run = 0, 
                             vals_to_read = ["cosmological_parameters--sigma_8", "intrinsic_alignment_parameters--a"],
                             mocks_dir = "/home/ruyi/cosmology/kcap_methods/tests/test_files",
                             mocks_name = "test_mock")

        golden_vals = {'cosmological_parameters--sigma_8': 0.8114996145348634, 'intrinsic_alignment_parameters--a': 0.974}
        self.assertEqual(vals, golden_vals)
    
    def test_read_covariance(self):
        covariance = km.get_covariance(mock_run = 1, 
                                       mocks_dir = "/home/ruyi/cosmology/kcap_methods/tests/test_files", 
                                       mocks_name = "test_mock")
        golden_cov = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        np.testing.assert_array_almost_equal(covariance, golden_cov, decimal = 5)

if __name__ == '__main__':
    unittest.main()