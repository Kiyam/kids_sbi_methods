import numpy as np
from . import deriv_run

#TODO Still need to figure out how to read the covariance matrix from kcap

def compute_fisher(fiducial_run, inv_covariance, x, deriv_params):
    """
    Computes the fisher matrix based on the params set, 
    e.g params = ["shear_xi_plus_omch2_deriv", "shear_xi_plus_sigma_8_deriv", "shear_xi_minus_omch2_deriv", "shear_xi_minus_sigma_8_deriv"]
    """
    fid_derivatives = deriv_run.get_values(mock_run = fiducial_run, vals_to_read = deriv_params)
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
    """
    # Fetch the fiducial means and the datavector
    data_vector_dict = deriv_run.get_values(mock_run = mock_run, vals_to_read = data_params)
    fid_vector_dict = deriv_run.get_values(mock_run = fiducial_run, vals_to_read = data_params)

    # Gets the first element from the fetched data_vector and calculates flattened size
    data_vector_length = list(data_vector_dict.values())[0].shape[0] * list(data_vector_dict.values())[0].shape[1]
    data_vector = np.zeros(shape = (len(data_params), data_vector_length))
    fiducial_vector = np.zeros(shape = (len(data_params), data_vector_length))

    #Store the data vector in a n by m array first, then flatten. This just makes the indexing process simpler.
    for i, (key, data_values) in enumerate(data_vector_dict.items()):
        if data_values.shape[0] * data_values.shape[1] == data_vector_length:
            data_vector[i] += data_values.flatten()
            fiducial_vector[i] += fid_vector_dict[key].flatten()
        else:
            raise Exception("Data vectors of different length between parameters")
    data_vector = data_vector.flatten()
    fiducial_vector = fiducial_vector.flatten()
    data_diff = data_vector - fiducial_vector

    # Fetch the derivative values at the fiducial params
    if isinstance(deriv_params[0], list) and len(deriv_params[0]) == 2:
        flat_deriv_params = [item for sublist in deriv_params for item in sublist]
    else:
        raise Exception("Expected a list of 2 element long sublists for the derivatives as we are varying more than 1 parameter at a time")
    fid_deriv_dict = deriv_run.get_values(mock_run = fiducial_run, vals_to_read = flat_deriv_params)

    deriv_matrix = np.zeros(shape = (len(deriv_params), len(x)))

    for i, grouped_deriv_params in enumerate(deriv_params):
        temp_deriv_array = np.zeros(shape = (len(deriv_params[0]), data_vector_length))
        for j, deriv_params in enumerate(grouped_deriv_params):
            temp_deriv_array[j] += fid_deriv_dict[deriv_params].flatten()
        deriv_matrix[i] += temp_deriv_array.flatten()
    
    # Now to do the matrix multiplications for score compression
    linear_score = np.dot(deriv_matrix, np.dot(inv_covariance, data_diff))

    if linear is True:
        return linear_score
    else:
        cov_deriv = deriv_run.get_values()
        cov_score = np.dot(np.transpose(data_diff), np.dot(inv_covariance, np.dot(cov_deriv, np.dot(inv_covariance, data_diff))))
        score = linear_score + cov_score
        return score

def write_file(input_array, file_location):
    outfile = open(file_location, 'w')
    # outfile.write('# t    s(t)\n')  # write table header
    for rows in input_array:
        outfile.write(rows + "\n")