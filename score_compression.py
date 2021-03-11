import numpy as np
from . import deriv_run

def compute_fisher(inv_covariance, x, params):
    """
    Computes the fisher matrix based on the params set, 
    e.g params = ["shear_xi_plus_omch2_deriv", "shear_xi_plus_sigma_8_deriv", "shear_xi_minus_omch2_deriv", "shear_xi_minus_sigma_8_deriv"]
    """
    derivatives = deriv_run.get_values(mock_run = 0, vals_to_read = params)
    term_1 = np.zeros(shape = (len(params), len(x)))
    for i, x_val in enumerate(x):
        for j, deriv_val in enumerate(params):
            term[j][i] = derivatives[deriv_val]

    term_3 = np.transpose(term_1)

    term_2 = np.dot(inv_covariance, term_3)
    fisher_matrix = np.dot(term_1, term_2)

    return fisher_matrix

def linear_score_compress(inv_covariance, x, params, data_params):
    # Find the first term, the transposed dmu/dt term
    derivatives = deriv_run.get_values(mock_run = 0, vals_to_read = params)
    data_vector = deriv_run.get_values(mock_run = 0, vals_to_read = data_params)
    fiducial_means = deriv_run.get_fiducial_means(mock_run = 0)
    term_1 = np.zeros(shape = (len(params), len(x)))
    for i, x_val in enumerate(x):
        for j, deriv_val in enumerate(params):
            term[j][i] = derivatives[deriv_val]
    # The n_d vector of d - mu
    term_3 = data_vector - fiducial_means
    # The matrix multiplication between inverted covariance and term_3
    term_2 = np.dot(inv_covariance, term_3)
    score = np.dot(term_1, term_2)
    return score

def compress_data(inv_covariance, data, fiducial_data, scales, sigma, priors):
    # Will use the derivatives and compressor setup earlier to do this. This is just a packaging of previous code.abs
    compressed_data = np.zeros(shape=(len(priors), len(priors[0])))
    for i, generated_data in enumerate(data):
        temp_score = linear_score_compress(data_vector = generated_data, fiducial_means = fiducial_data[i], inv_covariance = inv_covariance, scales = scales, fiducial_amp = priors[i][1], fiducial_slope = priors[i][0])
        compressed_data[i] = temp_score
    
    return compressed_data