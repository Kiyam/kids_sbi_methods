import numpy as np
import kcap_methods
import os
import errno

#TODO Still need to figure out how to read the covariance matrix from kcap

def compute_fisher(fiducial_run, deriv_params, data_params, mocks_dir = None, mocks_name = None):
    """
    Computes the fisher matrix based on the params set, 
    e.g params = ["shear_xi_plus_omch2_deriv", "shear_xi_plus_sigma_8_deriv", "shear_xi_minus_omch2_deriv", "shear_xi_minus_sigma_8_deriv"]
    """
    inv_covariance =  kcap_methods.get_inv_covariance(mock_run = fiducial_run, mocks_dir = mocks_dir, mocks_name = mocks_name)
    assert inv_covariance.shape[0] == inv_covariance.shape[1], "Square matrix for the inverse covariance not found"

    deriv_matrix = np.zeros(shape = (len(deriv_params), inv_covariance.shape[0]))
    for i, deriv_param in enumerate(deriv_params):
        deriv_vals_to_get = [data_param + '_' + deriv_param + '_deriv' for data_param in data_params]
        deriv_vector_dict = kcap_methods.get_values(mock_run = fiducial_run, vals_to_read = deriv_vals_to_get, mocks_dir = mocks_dir, mocks_name = mocks_name)
        deriv_vector = np.array([])
        for data_param in data_params:
            deriv_vector = np.append(deriv_vector, deriv_vector_dict[data_param+'_'+deriv_param+'_deriv'])
        
        deriv_matrix[i] += deriv_vector

    fisher_matrix = np.dot(deriv_matrix, np.dot(inv_covariance, np.transpose(deriv_matrix)))
    return fisher_matrix

def regular_likelihood_posterior(sim_number, fiducial_run, data_params, params, mocks_dir, mocks_name):
    """
    Calculates the regular posterior assuming Gaussian likelihood.
    """
    thetas = np.zeros(shape = (sim_number, len(params)))
    gaus_post = np.zeros(sim_number)
    likelihood = np.zeros(sim_number)
    fid_vector = get_fiducial_vector(fiducial_run = fiducial_run, data_params = data_params, mocks_dir = mocks_dir, mocks_name = mocks_name)
    inv_cov = kcap_methods.get_inv_covariance(mock_run = fiducial_run, mocks_dir = mocks_dir, mocks_name = mocks_name) #we do this hear just because it's parameter independent, so it's the same covariance for all parameter points

    for i in range(sim_number):
        param_dict = kcap_methods.get_params(mock_run = i, vals_to_read = params, mocks_dir = mocks_dir, mocks_name = mocks_name)
        thetas[i] += np.array([param_dict[param] for param in params])
        data_vector_dict = kcap_methods.get_values(mock_run = i, vals_to_read = data_params, mocks_dir = mocks_dir, mocks_name = mocks_name) # The datavector stored as a dict of 2 flattened numpy arrays
        data_vector = np.array([])
        for data_param in data_params:
            data_vector = np.append(data_vector, data_vector_dict[data_param])
        # Should now have a 1d datavector.

        data_diff = data_vector - fid_vector
        
        gaus_post[i] += np.exp(-0.5 * np.dot(np.transpose(data_diff), np.dot(inv_cov, data_diff)))
        likelihood[i] += kcap_methods.get_likelihood(mock_run = i, like_name = "2x2pt_like_like", mocks_dir = mocks_dir, mocks_name = mocks_name)
    
    return thetas, gaus_post, likelihood    

def get_fiducial_deriv(fiducial_run, deriv_params, data_params, mocks_dir = None, mocks_name = None):
    for i, deriv_param in enumerate(deriv_params):
        if i == 0:
            deriv_vals_to_get = [data_param + '_' + deriv_param + '_deriv' for data_param in data_params]
            deriv_vector_dict = kcap_methods.get_values(mock_run = fiducial_run, vals_to_read = deriv_vals_to_get, mocks_dir = mocks_dir, mocks_name = mocks_name)
            deriv_vector = np.array([])
            for data_deriv_param in deriv_vals_to_get:
                deriv_vector = np.append(deriv_vector, deriv_vector_dict[data_deriv_param])
            deriv_matrix = np.zeros(shape = (len(deriv_params), len(deriv_vector)))
        else:
            deriv_vals_to_get = [data_param + '_' + deriv_param + '_deriv' for data_param in data_params]
            deriv_vector_dict = kcap_methods.get_values(mock_run = fiducial_run, vals_to_read = deriv_vals_to_get, mocks_dir = mocks_dir, mocks_name = mocks_name)
            deriv_vector = np.array([])
            for data_deriv_param in deriv_vals_to_get:
                deriv_vector = np.append(deriv_vector, deriv_vector_dict[data_deriv_param])
        
        deriv_matrix[i] += deriv_vector
    
    return deriv_matrix

def get_fiducial_vector(fiducial_run, data_params, mocks_dir = None, mocks_name = None):
    fid_vector_dict = kcap_methods.get_values(mock_run = fiducial_run, vals_to_read = data_params, mocks_dir = mocks_dir, mocks_name = mocks_name)
    fid_vector = np.array([])
    for data_param in data_params:
        fid_vector = np.append(fid_vector, fid_vector_dict[data_param])
    
    return fid_vector

def get_fiducial_cov_deriv(fiducial_run, deriv_matrix, deriv_params, mocks_dir = None, mocks_name = None):
    cov_tensor_shape = list(deriv_matrix.shape)
    cov_tensor_shape.append(cov_tensor_shape[-1])
    cov_deriv_tensor = np.zeros(shape = cov_tensor_shape)
    for i, deriv_param in enumerate(deriv_params):
        cov_deriv = kcap_methods.get_covariance(mock_run = fiducial_run, which_cov = deriv_param, mocks_dir = mocks_dir, mocks_name = mocks_name)
        cov_deriv_tensor[i] += cov_deriv
    
    return cov_deriv_tensor

def score_compress(mock_run, fid_vector, deriv_matrix, data_params, cov_deriv_tensor = None, mocks_dir = None, mocks_name = None):
    """
    General Score compression

    data_params: These are the params that will be joined together to form the full datavector. It's important that these are written in the list in the right order
    deriv_params: A list of the variables that derivatives have been taken against. 
    """
    # Fetch the fiducial means and the datavector
    data_vector_dict = kcap_methods.get_values(mock_run = mock_run, vals_to_read = data_params, mocks_dir = mocks_dir, mocks_name = mocks_name) # The datavector stored as a dict of 2 flattened numpy arrays
    data_vector = np.array([])
    for data_param in data_params:
        data_vector = np.append(data_vector, data_vector_dict[data_param])
    # Should now have a 1d datavector.

    data_diff = data_vector - fid_vector  
    inv_covariance =  kcap_methods.get_inv_covariance(mock_run = mock_run, mocks_dir = mocks_dir, mocks_name = mocks_name)
    # Now to do the matrix multiplications for score compression
    linear_score = np.dot(deriv_matrix, np.dot(inv_covariance, data_diff))
    if cov_deriv_tensor is None:
        print("Linear Score Compression Done!")
        return linear_score
    else:
        cov_score = np.dot(np.dot(np.transpose(data_diff), inv_covariance), np.transpose(np.dot(cov_deriv_tensor, np.dot(inv_covariance, data_diff))))       
        score = linear_score + cov_score
        print("Non-Linear Score Compression Done!")
        return score

def do_compression(sim_number, fiducial_run, data_run, deriv_params, data_params, theta_names, linear = True, mocks_dir = None, mocks_name = None):
    """
    Do compression, returning 3 arrays. 
    theta_names - expected in the full cosmosis naming chain, e.g. cosmological_parameters--omch2
    """
    fid_vector = get_fiducial_vector(fiducial_run = fiducial_run, data_params = data_params, mocks_dir = mocks_dir, mocks_name = mocks_name)
    deriv_matrix = get_fiducial_deriv(fiducial_run = fiducial_run, deriv_params = deriv_params, data_params = data_params, mocks_dir = mocks_dir, mocks_name = mocks_name)
    if linear is not True:
        cov_deriv_tensor = get_fiducial_cov_deriv(fiducial_run = fiducial_run, deriv_matrix = deriv_matrix, deriv_params = deriv_params, mocks_dir = mocks_dir, mocks_name = mocks_name)
    else:
        cov_deriv_tensor = None

    compressed_data = np.zeros(shape = (sim_number, len(deriv_params)))
    thetas = np.zeros(shape = (sim_number, len(deriv_params)))

    for i in range(sim_number):
        score = score_compress(mock_run = i, fid_vector = fid_vector, deriv_matrix = deriv_matrix, data_params = data_params, cov_deriv_tensor = cov_deriv_tensor, mocks_dir = mocks_dir, mocks_name = mocks_name)
        compressed_data[i] += score

        theta = kcap_methods.get_params(mock_run = i, vals_to_read = theta_names, mocks_dir = mocks_dir, mocks_name = mocks_name)
        theta = np.array(list(theta.values()))
        thetas[i] += theta
    fid_data = compressed_data[data_run]
    print("Compression finished!")
    return compressed_data, thetas, fid_data
    
def write_file(input_array, file_location, file_name):
    outfile = file_location + '/' + file_name + '.dat'    
    if not os.path.exists(os.path.dirname(outfile)):
        try:
            os.makedirs(os.path.dirname(outfile))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    np.savetxt(outfile, input_array)
    print("File succesfully saved as %s" % str(outfile))

if __name__ == "__main__":
    mocks_dir = "/home/ruyi_wsl/kcap_output/kids_1000_mocks_trial_5"
    compressed_data, thetas, fid_data =  do_compression(sim_number = 4001, fiducial_run = 4001, data_run = 4000,
                                                            deriv_params = ['omega_m', 'sigma_8'], 
                                                            data_params = ['shear_xi_plus_binned', 'shear_xi_minus_binned'],
                                                            theta_names = ['cosmological_parameters--omega_m', 'cosmological_parameters--sigma_8'],
                                                            linear = True, 
                                                            mocks_dir = mocks_dir,
                                                            mocks_name = "kids_1000_cosmology")
    # compressed_data, thetas, fid_data =  do_compression(sim_number = 4001, fiducial_run = 4001, data_run = 4000,
    #                                                         deriv_params = ['omega_m', 's_8'], 
    #                                                         data_params = ['shear_xi_plus_binned', 'shear_xi_minus_binned'],
    #                                                         theta_names = ['cosmological_parameters--omega_m', 'cosmological_parameters--s_8'],
    #                                                         linear = True, 
    #                                                         mocks_dir = mocks_dir,
    #                                                         mocks_name = "kids_1000_cosmology")
    fisher_matrix = compute_fisher(fiducial_run = 4001, 
                                   deriv_params = ['omega_m', 'sigma_8'], 
                                   data_params = ['shear_xi_plus_binned', 'shear_xi_minus_binned'], 
                                   mocks_dir = mocks_dir,
                                   mocks_name = "kids_1000_cosmology")
    # fisher_matrix = compute_fisher(fiducial_run = 4001, 
    #                                deriv_params = ['omega_m', 's_8'], 
    #                                data_params = ['shear_xi_plus_binned', 'shear_xi_minus_binned'], 
    #                                mocks_dir = mocks_dir,
    #                                mocks_name = "kids_1000_cosmology")

    file_loc_compress = mocks_dir + '/compressed_data'
    write_file(input_array = compressed_data, file_location = file_loc_compress, file_name = 'compressed_data')
    write_file(input_array = thetas, file_location = file_loc_compress, file_name = 'thetas')
    write_file(input_array = fid_data, file_location = file_loc_compress, file_name = 'fid_data')
    write_file(input_array = fisher_matrix, file_location = file_loc_compress, file_name = 'fisher_matrix')

    thetas, post, likelihood = regular_likelihood_posterior(sim_number = 4002, fiducial_run = 4001, 
                                                data_params = ['shear_xi_plus_binned', 'shear_xi_minus_binned'], 
                                                params = ['cosmological_parameters--omega_m', 'cosmological_parameters--s_8'], 
                                                mocks_dir = mocks_dir,
                                                mocks_name = "kids_1000_cosmology")

    file_loc_conventional = mocks_dir + '/conventional_likelihood'    
    write_file(input_array = thetas, file_location = file_loc_conventional, file_name = 'thetas')
    write_file(input_array = post, file_location = file_loc_conventional, file_name = 'posterior')
    write_file(input_array = likelihood, file_location = file_loc_conventional, file_name = 'likelihood')