import numpy as np
import kcap_methods

#TODO Still need to figure out how to read the covariance matrix from kcap

def compute_fisher(mock_run, fiducial_run, deriv_params, data_params):
    """
    Computes the fisher matrix based on the params set, 
    e.g params = ["shear_xi_plus_omch2_deriv", "shear_xi_plus_sigma_8_deriv", "shear_xi_minus_omch2_deriv", "shear_xi_minus_sigma_8_deriv"]
    """
    inv_covariance =  kcap_methods.get_inv_covariance(mock_run = mock_run)
    assert inv_covariance.shape[0] == inv_covariance.shape[1], "Square matrix for the inverse covariance not found"

    deriv_matrix = np.zeros(shape = (len(deriv_params), inv_covariance.shape[0]))
    for i, deriv_param in enumerate(deriv_params):
        deriv_vals_to_get = [data_param + '_' + deriv_param + '_deriv' for data_param in data_params]
        deriv_vector_dict = kcap_methods.get_values(mock_run = fiducial_run, vals_to_read = deriv_vals_to_get)
        deriv_vector = np.array([])
        for data_param in data_params:
            deriv_vector = np.append(deriv_vector, deriv_vector_dict[data_param+'_'+deriv_param+'_deriv'])
        
        deriv_matrix[i] += deriv_vector

    fisher_matrix = np.dot(deriv_matrix, np.dot(inv_covariance, np.transpose(deriv_matrix)))
    return fisher_matrix

def get_fiducial_deriv(fiducial_run, deriv_params, data_params):
    for i, deriv_param in enumerate(deriv_params):
        if i == 0:
            deriv_vals_to_get = [data_param + '_' + deriv_param + '_deriv' for data_param in data_params]
            deriv_vector_dict = kcap_methods.get_values(mock_run = fiducial_run, vals_to_read = deriv_vals_to_get)
            deriv_vector = np.array([])
            for data_deriv_param in deriv_vals_to_get:
                deriv_vector = np.append(deriv_vector, deriv_vector_dict[data_deriv_param])
            deriv_matrix = np.zeros(shape = (len(deriv_params), len(deriv_vector)))
        else:
            deriv_vals_to_get = [data_param + '_' + deriv_param + '_deriv' for data_param in data_params]
            deriv_vector_dict = kcap_methods.get_values(mock_run = fiducial_run, vals_to_read = deriv_vals_to_get)
            deriv_vector = np.array([])
            for data_deriv_param in deriv_vals_to_get:
                deriv_vector = np.append(deriv_vector, deriv_vector_dict[data_deriv_param])
        
        deriv_matrix[i] += deriv_vector
    
    return deriv_matrix

def get_fiducial_vector(fiducial_run, data_params):
    fid_vector_dict = kcap_methods.get_values(mock_run = fiducial_run, vals_to_read = data_params)
    fid_vector = np.array([])
    for data_param in data_params:
        fid_vector = np.append(fid_vector, fid_vector_dict[data_param])
    
    return fid_vector

def get_fiducial_cov_deriv(fiducial_run, deriv_matrix, deriv_params):
    cov_tensor_shape = list(deriv_matrix.shape)
    cov_tensor_shape.append(cov_tensor_shape[-1])
    cov_deriv_tensor = np.zeros(shape = cov_tensor_shape)
    for i, deriv_param in enumerate(deriv_params):
        cov_deriv = kcap_methods.get_covariance(mock_run = fiducial_run, which_cov = deriv_param)
        cov_deriv_tensor[i] += cov_deriv
    
    return cov_deriv_tensor

def score_compress(mock_run, fid_vector, deriv_matrix, data_params, cov_deriv_tensor = None):
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

    data_diff = data_vector - fid_vector  
    inv_covariance =  kcap_methods.get_inv_covariance(mock_run = mock_run)

    # Now to do the matrix multiplications for score compression
    linear_score = np.dot(deriv_matrix, np.dot(inv_covariance, data_diff))
    if cov_deriv_tensor is None:
        return linear_score
    else:
        cov_score = np.dot(np.dot(np.transpose(data_diff), inv_covariance), np.transpose(np.dot(cov_deriv_tensor, np.dot(inv_covariance, data_diff))))       
        score = linear_score + cov_score
        return score

def do_compression(sim_number, fiducial_run, deriv_params, data_params, theta_names):
    """
    Do compression, returning 3 arrays. 
    theta_names - expected in the full cosmosis naming chain, e.g. cosmological_parameters--omch2
    """
    fid_vector = get_fiducial_vector(fiducial_run = fiducial_run, data_params = data_params)
    deriv_matrix = get_fiducial_deriv(fiducial_run = fiducial_run, deriv_params = deriv_params, data_params = data_params)
    cov_deriv_tensor = get_fiducial_cov_deriv(fiducial_run = fiducial_run, deriv_matrix = deriv_matrix, deriv_params = deriv_params)

    compressed_data = np.zeros(shape = (sim_number, len(deriv_params)))
    thetas = np.zeros(shape = (sim_number, len(deriv_params)))

    for i in range(sim_number):
        score = score_compress(mock_run = i, fid_vector = fid_vector, deriv_matrix = deriv_matrix, data_params = data_params, cov_deriv_tensor = cov_deriv_tensor)
        compressed_data[i] += score

        theta = kcap_methods.get_params(mock_run = i, vals_to_read = theta_names)
        theta = np.array(list(theta.values()))
        thetas[i] += theta

    fid_data = compressed_data[int(sim_number/2)] #TODO To pick the "real" data, just need another fiducial dataset to get the compressed values
    return compressed_data, thetas, fid_data
    
def write_file(input_array, file_location):
    outfile = file_location + '.dat'    
    np.savetxt(outfile, input_array)

if __name__ == "__main__":
    compressed_data, thetas, fid_data =  do_compression(sim_number = 125, fiducial_run = 125, deriv_params = ['omega_m', 'sigma_8', 'a'], 
                                                                                          data_params = ['shear_xi_plus_binned', 'shear_xi_minus_binned'],
                                                                                          theta_names = ['cosmological_parameters--omega_m', 'cosmological_parameters--sigma_8', 'intrinsic_alignment_parameters--a'])
    write_file(input_array = compressed_data, file_location = '/home/ruyi_wsl/compressed_data/compressed_data')
    write_file(input_array = thetas, file_location = '/home/ruyi_wsl/compressed_data/thetas')
    write_file(input_array = fid_data, file_location = '/home/ruyi_wsl/compressed_data/fid_data')