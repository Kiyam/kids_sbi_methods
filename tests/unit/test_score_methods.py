import unittest
import numpy as np
import kcap_methods as km
import score_compression as sc

class TestScoreMethods(unittest.TestCase):
    def test_score_compression(self):
        data_vector = np.array([6, 5, 4])
        fid_vector = np.array([5.8, 4.7, 3.9])
        deriv_matrix = np.array([[0.4,0.7,0.9],[2.1,2.6,2.7]])
        inv_covariance = np.array([[-0.3, -0.2, 0.9],[0.7, -0.2, -1.1],[-0.60, 0.6, 0.8]])
        score = sc.score_compress(data_vector = data_vector,
                                  fid_vector = fid_vector,
                                  deriv_matrix = deriv_matrix,
                                  inv_covariance = inv_covariance)
        golden_score = np.array([0.093, 0.237])
        np.testing.assert_array_almost_equal(score, golden_score, decimal = 5)
    
    def test_read_fiducial_derivatives(self):
        derivs = sc.get_fiducial_deriv(fiducial_run = 1,
                                       deriv_params = ['deriv_1', 'deriv_2'],
                                       data_params = ['data_1', 'data_2'],
                                       mocks_dir = "/home/ruyi/cosmology/kcap_methods/tests/test_files", 
                                       mocks_name = "test_mock",
                                       bin_order = ['bin_1_1', 'bin_2_1', 'bin_3_2', 'bin_4_3'])
        
        golden_deriv = np.array([[111, 121, 131, 141, 151,
                                  121, 131, 141, 151, 161,
                                  131, 141, 151, 161, 171,
                                  141, 151, 161, 171, 181,
                                  211, 221, 231, 241, 251,
                                  122, 132, 142, 152, 162,
                                  132, 142, 152, 162, 172,
                                  142, 152, 162, 172, 182],
                                 [112, 122, 132, 142, 152,
                                  122, 132, 142, 152, 162,
                                  132, 142, 152, 162, 172,
                                  142, 152, 162, 172, 182,
                                  212, 222, 232, 242, 252,
                                  222, 232, 242, 252, 262,
                                  232, 242, 252, 262, 272,
                                  242, 252, 262, 272, 282]])
        
        np.testing.assert_array_almost_equal(derivs, golden_deriv, decimal = 5)
    
    def test_computer_fisher(self):
        fisher_matrix = sc.compute_fisher(fiducial_run = 1,
                                          deriv_params = ['deriv_1', 'deriv_2'],
                                          data_params = ['data_1', 'data_2'],
                                          mocks_dir = "/home/ruyi/cosmology/kcap_methods/tests/test_files", 
                                          mocks_name = "test_mock",
                                          bin_order = ['bin_1_1', 'bin_2_1', 'bin_3_2', 'bin_4_3'])

        golden_deriv = np.array([[111, 121, 131, 141, 151,
                                  121, 131, 141, 151, 161,
                                  131, 141, 151, 161, 171,
                                  141, 151, 161, 171, 181,
                                  211, 221, 231, 241, 251,
                                  122, 132, 142, 152, 162,
                                  132, 142, 152, 162, 172,
                                  142, 152, 162, 172, 182],
                                 [112, 122, 132, 142, 152,
                                  122, 132, 142, 152, 162,
                                  132, 142, 152, 162, 172,
                                  142, 152, 162, 172, 182,
                                  212, 222, 232, 242, 252,
                                  222, 232, 242, 252, 262,
                                  232, 242, 252, 262, 272,
                                  242, 252, 262, 272, 282]])
        
        golden_inv_covariance = km.get_inv_covariance(mock_run = 1,
                                                       mocks_dir = "/home/ruyi/cosmology/kcap_methods/tests/test_files", 
                                                       mocks_name = "test_mock")
        
        golden_fisher = np.dot(golden_deriv, np.dot(golden_inv_covariance, np.transpose(golden_deriv)))

        np.testing.assert_array_almost_equal(fisher_matrix, golden_fisher, decimal = 5)

if __name__ == '__main__':
    unittest.main()