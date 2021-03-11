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

def linear_score_compress(mock_run, fiducial_run, inv_covariance, x, deriv_params, data_params):
    # Find the first term, the transposed dmu/dt term
    data_vector_dict = deriv_run.get_values(mock_run = mock_run, vals_to_read = data_params)
    fid_derivatives = deriv_run.get_values(mock_run = fiducial_run, vals_to_read = deriv_params)
    fid_vector_dict = deriv_run.get_values(mock_run = fiducial_run, vals_to_read = data_params)
    cov_deriv = deriv_run.get_values()

    # Gets the first element from the fetched data_vector and calculates flattened size
    data_vector_length = list(data_vector_dict.values())[0].shape[0] * list(data_vector_dict.values())[0].shape[1]
    data_vector = np.zeros(shape = (len(data_params), data_vector_length))
    fiducial_vector = np.zeros(shape = (len(data_params), data_vector_length))

    for i, (key, data_values) in enumerate(data_vector_dict.items()):
        if data_values.shape[0] * data_values.shape[1] == data_vector_length:
            data_vector[i] += data_values.flatten()
            fiducial_vector[i] += fid_vector_dict[key].flatten()
        else:
            raise Exception("Data vectors of different length between parameters")
    data_diff = data_vector - fiducial_vector
    deriv_matrix = np.zeros(shape = (len(deriv_params), len(x)))
    for i, x_val in enumerate(x):
        for j, deriv_val in enumerate(deriv_params):
            deriv_matrix[j][i] += fid_derivatives[deriv_val]
    linear_score = np.dot(deriv_matrix, np.dot(inv_covariance, data_diff))

    cov_score = np.dot(np.transpose(data_diff), np.dot(inv_covariance, np.dot(cov_deriv, np.dot(inv_covariance, data_diff))))

    score = linear_score + cov_score
    return score