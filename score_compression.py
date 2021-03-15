import numpy as np
from . import kcap_methods

#TODO Still need to figure out how to read the covariance matrix from kcap

def compute_fisher(fiducial_run, inv_covariance, x, deriv_params):
    """
    Computes the fisher matrix based on the params set, 
    e.g params = ["shear_xi_plus_omch2_deriv", "shear_xi_plus_sigma_8_deriv", "shear_xi_minus_omch2_deriv", "shear_xi_minus_sigma_8_deriv"]
    """
    fid_derivatives = kcap_methods.get_values(mock_run = fiducial_run, vals_to_read = deriv_params)
    deriv_vector_length = list(fid_derivatives.values())[0].shape[0] * list(fid_derivatives.values())[0].shape[1]
    deriv_vector = np.zeros(shape = (len(deriv_params), deriv_vector_length))
    for i, (key, data_values) in enumerate(fid_derivatives.items()):
        if data_values.shape[0] * data_values.shape[1] == deriv_vector_length:
            deriv_vector[i] += data_values.flatten()
        else:
            raise Exception("Data vectors of different length between parameters")
    deriv_matrix = np.zeros(shape = (len(deriv_params), len(x)))
    for i, x_val in enumerate(x):
        for j, deriv_val in enumerate(deriv_params):
            deriv_matrix[j][i] += deriv_vector[j]

    fisher_matrix = np.dot(deriv_matrix, np.dot(inv_covariance, np.transpose(deriv_matrix)))
    return fisher_matrix

def score_compress(mock_run, fiducial_run, inv_covariance, x, deriv_params, data_params, linear = True):
    """
    General Score compression

    data_params: These are the params that will be joined together to form the full datavector. It's important that these are written in the list in the right order
    deriv_params: A list of the variables that derivatives have been taken against. 
    """
    # Fetch the fiducial means and the datavector
    data_vector_dict = kcap_methods.get_values(mock_run = mock_run, vals_to_read = data_params) # The datavector stored as a dict of 2 flattened numpy arrays
    data_vector = np.array([])
    for data_param in data_params:
        data_vector = np.append(data_vector, data_vector_dict[data_param])
    # Should now have a 1d datavector.

    fid_vector_dict = kcap_methods.get_values(mock_run = fiducial_run, vals_to_read = data_params)
    fid_vector = np.array([])
    for data_param in data_params:
        fid_vector = np.append(fid_vector, fid_vector_dict[data_param])
    
    data_diff = data_vector - fid_vector
    
    deriv_matrix = np.zeros(shape = (len(deriv_params), len(data_vector)))
    for i, deriv_param in enumerate(deriv_params):
        deriv_vals_to_get = [data_param + '_' + deriv_param + '_deriv' for data_param in data_params]
        deriv_vector_dict = kcap_methods.get_values(mock_run = fiducial_run, vals_to_read = deriv_vals_to_get)
        deriv_vector = np.array([])
        for data_param in data_params:
            deriv_vector = np.append(deriv_vector, deriv_vector_dict[data_param])
        
        deriv_matrix[i] += deriv_vector
    
    inv_covariance =  kcap_methods.get_inv_covariance(mock_run = mock_run)

    # Now to do the matrix multiplications for score compression
    linear_score = np.dot(deriv_matrix, np.dot(inv_covariance, data_diff))

    if linear is True:
        return linear_score
    else:
        deriv_matrix_shape = deriv_matrix.shape
        cov_tensor_shape = [len(deriv_params), deriv_matrix_shape[0], deriv_matrix_shape[1]]
        cov_deriv_tensor = np.zeros(shape = cov_tensor_shape)
        for i, deriv_param in enumerate(deriv_params):
            cov_deriv = kcap_methods.get_covariance(mock_run = fiducial_run, which_cov = deriv_param)

        cov_score = np.dot(np.transpose(data_diff), np.dot(inv_covariance, np.dot(cov_deriv, np.dot(inv_covariance, data_diff))))
        score = linear_score + cov_score
        return score

def write_file(input_array, file_location):
    outfile = open(file_location, 'w')
    # outfile.write('# t    s(t)\n')  # write table header
    for rows in input_array:
        outfile.write(rows + "\n")