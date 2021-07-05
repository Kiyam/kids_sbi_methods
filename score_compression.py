import numpy as np
import os
import errno
import kcap_methods

#TODO Still need to figure out how to read the covariance matrix from kcap

def compute_fisher(fiducial_run, deriv_params, data_params, mocks_dir = None, mocks_name = None, bin_order = None):
    """
    Computes the fisher matrix based on the params set, 
    e.g params = ["shear_xi_plus_omch2_deriv", "shear_xi_plus_sigma_8_deriv", "shear_xi_minus_omch2_deriv", "shear_xi_minus_sigma_8_deriv"]
    """
    # inv_covariance = kcap_methods.get_inv_covariance(mock_run = fiducial_run, mocks_dir = mocks_dir, mocks_name = mocks_name)
    # assert inv_covariance.shape[0] == inv_covariance.shape[1], "Square matrix for the inverse covariance not found"
    
    # symmetrised_inv_covariance = kcap_methods.get_inv_covariance(mock_run = fiducial_run, which_cov = "symmetrised", mocks_dir = mocks_dir, mocks_name = mocks_name)
    # cholesky_inv_covariance = kcap_methods.get_inv_covariance(mock_run = fiducial_run, which_cov = "cholesky", mocks_dir = mocks_dir, mocks_name = mocks_name)
    eigen_inv_covariance = kcap_methods.get_inv_covariance(mock_run = fiducial_run, which_cov = "eigen", mocks_dir = mocks_dir, mocks_name = mocks_name)

    deriv_matrix = np.zeros(shape = (len(deriv_params), eigen_inv_covariance.shape[0]))
    for i, deriv_param in enumerate(deriv_params):
        deriv_vals_to_get = [data_param + '_' + deriv_param + '_deriv' for data_param in data_params]
        deriv_vector_dict = kcap_methods.get_values(mock_run = fiducial_run, vals_to_read = deriv_vals_to_get, mocks_dir = mocks_dir, mocks_name = mocks_name, bin_order = bin_order)
        deriv_vector = np.array([])
        for data_param in data_params:
            deriv_vector = np.append(deriv_vector, deriv_vector_dict[data_param+'_'+deriv_param+'_deriv'])
        
        deriv_matrix[i] += deriv_vector

    # fisher_matrix = np.dot(deriv_matrix, np.dot(inv_covariance, np.transpose(deriv_matrix)))
    # syminv_fisher_matrix = np.dot(deriv_matrix, np.dot(symmetrised_inv_covariance, np.transpose(deriv_matrix)))
    # cholesky_fisher_matrix = np.dot(deriv_matrix, np.dot(cholesky_inv_covariance, np.transpose(deriv_matrix)))
    eigen_fisher_matrix = np.dot(deriv_matrix, np.dot(eigen_inv_covariance, np.transpose(deriv_matrix)))
    return eigen_fisher_matrix

def regular_likelihood_posterior(sim_number, fiducial_run, data_params, params, mocks_dir = None, mocks_name = None, noisey_data = False):
    """
    Calculates the regular posterior assuming Gaussian likelihood.
    """
    thetas = np.zeros(shape = (sim_number, len(params)))
    gaus_post = np.zeros(sim_number)
    likelihood = np.zeros(sim_number)
    fid_vector = get_fiducial_vector(fiducial_run = fiducial_run, data_params = data_params, mocks_dir = mocks_dir, mocks_name = mocks_name)
    eigen_inv_covariance = kcap_methods.get_inv_covariance(mock_run = fiducial_run, which_cov = "eigen", mocks_dir = mocks_dir, mocks_name = mocks_name)
    # inv_cov = kcap_methods.get_inv_covariance(mock_run = fiducial_run, mocks_dir = mocks_dir, mocks_name = mocks_name) #we do this here just because it's parameter independent, so it's the same covariance for all parameter points

    for i in range(sim_number):
        param_dict = kcap_methods.get_params(mock_run = i, vals_to_read = params, mocks_dir = mocks_dir, mocks_name = mocks_name)
        thetas[i] += np.array([param_dict[param] for param in params])
        # Fetch the datavector
        if noisey_data is True:
            data_vector = kcap_methods.get_noisey_data(mock_run = i, mocks_dir = mocks_dir, mocks_name = mocks_name)
        else:
            data_vector_dict = kcap_methods.get_values(mock_run = i, vals_to_read = data_params, mocks_dir = mocks_dir, mocks_name = mocks_name) # The datavector stored as a dict of 2 flattened numpy arrays
            data_vector = np.array([])
            for data_param in data_params:
                data_vector = np.append(data_vector, data_vector_dict[data_param])
            # Should now have a 1d datavector.

        data_diff = data_vector - fid_vector
        
        gaus_post[i] += np.exp(-0.5 * np.dot(np.transpose(data_diff), np.dot(eigen_inv_covariance, data_diff)))
        likelihood[i] += kcap_methods.get_likelihood(mock_run = i, like_name = "loglike_like", mocks_dir = mocks_dir, mocks_name = mocks_name)
    
    return thetas, gaus_post, likelihood

def compute_fisher_likelihood(sim_number, fiducial_run, data_params, params, fisher, mocks_dir, mocks_name):
    """
    Calculates the regular posterior assuming Gaussian likelihood.
    """
    thetas = np.zeros(shape = (sim_number, len(params)))
    fisher_likelihood = np.zeros(sim_number)
    fid_theta_dict = kcap_methods.get_params(mock_run = fiducial_run, vals_to_read = params, mocks_dir = mocks_dir, mocks_name = mocks_name)
    fid_theta = np.array([fid_theta_dict[param] for param in params])

    for i in range(sim_number):
        param_dict = kcap_methods.get_params(mock_run = i, vals_to_read = params, mocks_dir = mocks_dir, mocks_name = mocks_name)
        theta = np.array([param_dict[param] for param in params])
        thetas[i] += theta
        theta_diff = theta - fid_theta      
        fisher_likelihood[i] += np.exp(-0.5 * np.dot(np.transpose(theta_diff), np.dot(fisher, theta_diff)))
    
    return thetas, fisher_likelihood     

def get_fiducial_deriv(fiducial_run, deriv_params, data_params, mocks_dir = None, mocks_name = None, bin_order = None):
    for i, deriv_param in enumerate(deriv_params):
        if i == 0:
            deriv_vals_to_get = [data_param + '_' + deriv_param + '_deriv' for data_param in data_params]
            deriv_vector_dict = kcap_methods.get_values(mock_run = fiducial_run, vals_to_read = deriv_vals_to_get, mocks_dir = mocks_dir, mocks_name = mocks_name, bin_order = bin_order)
            deriv_vector = np.array([])
            for data_deriv_param in deriv_vals_to_get:
                deriv_vector = np.append(deriv_vector, deriv_vector_dict[data_deriv_param])
            deriv_matrix = np.zeros(shape = (len(deriv_params), len(deriv_vector)))
        else:
            deriv_vals_to_get = [data_param + '_' + deriv_param + '_deriv' for data_param in data_params]
            deriv_vector_dict = kcap_methods.get_values(mock_run = fiducial_run, vals_to_read = deriv_vals_to_get, mocks_dir = mocks_dir, mocks_name = mocks_name, bin_order = bin_order)
            deriv_vector = np.array([])
            for data_deriv_param in deriv_vals_to_get:
                deriv_vector = np.append(deriv_vector, deriv_vector_dict[data_deriv_param])
        
        deriv_matrix[i] += deriv_vector
    
    return deriv_matrix

def get_fiducial_vector(fiducial_run, data_params, mocks_dir = None, mocks_name = None, bin_order = None):
    fid_vector_dict = kcap_methods.get_values(mock_run = fiducial_run, vals_to_read = data_params, mocks_dir = mocks_dir, mocks_name = mocks_name, bin_order = bin_order)
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

def score_compress(data_vector, fid_vector, deriv_matrix, inv_covariance, cov_deriv_tensor = None):
    """
    General Score compression
    """
    data_diff = data_vector - fid_vector  
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

def do_compression(sim_number, fiducial_run, data_run, deriv_params, data_params, theta_names, linear = True, mocks_dir = None, mocks_name = None, noisey_data = True):
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
    
    eigen_inv_covariance = kcap_methods.get_inv_covariance(mock_run = fiducial_run, which_cov = "eigen", mocks_dir = mocks_dir, mocks_name = mocks_name)
    eigen_compressed_data = np.zeros(shape = (sim_number, len(deriv_params)))
    thetas = np.zeros(shape = (sim_number, len(deriv_params)))

    for i in range(sim_number):
        # Fetch the datavector
        if noisey_data is True and i != data_run:
            data_vector = kcap_methods.get_noisey_data(mock_run = i, mocks_dir = mocks_dir, mocks_name = mocks_name)
        else:
            data_vector_dict = kcap_methods.get_values(mock_run = i, vals_to_read = data_params, mocks_dir = mocks_dir, mocks_name = mocks_name) # The datavector stored as a dict of 2 flattened numpy arrays
            data_vector = np.array([])
            for data_param in data_params:
                data_vector = np.append(data_vector, data_vector_dict[data_param])
            # Should now have a 1d datavector.
        eigen_score = score_compress(data_vector = data_vector, fid_vector = fid_vector, deriv_matrix = deriv_matrix, inv_covariance = eigen_inv_covariance, cov_deriv_tensor = cov_deriv_tensor)
        
        eigen_compressed_data[i] += eigen_score
        
        theta = kcap_methods.get_params(mock_run = i, vals_to_read = theta_names, mocks_dir = mocks_dir, mocks_name = mocks_name)
        theta = np.array(list(theta.values()))
        thetas[i] += theta
        
    eigen_fid_data = eigen_compressed_data[data_run]
    
    print("Compression finished!")
    return eigen_compressed_data, thetas, eigen_fid_data
    
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

def main(deriv_params, data_params, theta_names, mocks_dir, mocks_name, sim_number, compressed_name = 'compressed_data', linear_compression = True, noisey_data = True):
    file_loc = mocks_dir + '/' + compressed_name
    eigen_compressed_data, thetas, eigen_fid_data = do_compression(sim_number = sim_number+1, fiducial_run = sim_number+1, data_run = sim_number,
                                                                    deriv_params = deriv_params, 
                                                                    data_params = data_params,
                                                                    theta_names = theta_names,
                                                                    linear = linear_compression, 
                                                                    mocks_dir = mocks_dir,
                                                                    mocks_name = mocks_name,
                                                                    noisey_data = noisey_data)

    eigen_fisher_matrix = compute_fisher(fiducial_run = sim_number+1, 
                                        deriv_params = deriv_params, 
                                        data_params = data_params, 
                                        mocks_dir = mocks_dir,
                                        mocks_name = mocks_name)

    write_file(input_array = thetas, file_location = file_loc, file_name = 'thetas')
    write_file(input_array = eigen_compressed_data, file_location = file_loc, file_name = 'eigen_compressed_data')
    write_file(input_array = eigen_fid_data, file_location = file_loc, file_name = 'eigen_fid_data')
    write_file(input_array = eigen_fisher_matrix, file_location = file_loc, file_name = 'eigen_fisher_matrix')       
    
    thetas, post, likelihood = regular_likelihood_posterior(sim_number = sim_number+1, fiducial_run = sim_number+1, 
                                                data_params = data_params, 
                                                params = theta_names, 
                                                mocks_dir = mocks_dir,
                                                mocks_name = mocks_name,
                                                noisey_data = noisey_data)
    
    write_file(input_array = post, file_location = file_loc, file_name = 'eigen_posterior')
    write_file(input_array = likelihood, file_location = file_loc, file_name = 'eigen_likelihood')

if __name__ == "__main__":    
    # main(deriv_params = ['cosmological_parameters--omega_m', 'cosmological_parameters--sigma_8',
    #                      'intrinsic_alignment_parameters--a', 'cosmological_parameters--n_s', 
    #                      'cosmological_parameters--h0', 'halo_model_parameters--a'], 
    #      data_params = ['theory'], 
    #      theta_names = ['cosmological_parameters--omega_m', 'cosmological_parameters--sigma_8',
    #                      'intrinsic_alignment_parameters--a', 'cosmological_parameters--n_s', 
    #                      'cosmological_parameters--h0', 'halo_model_parameters--a'], 
    #      mocks_dir = "/mnt/Node-Temp/cosmology/kcap_output/kids_1000_mocks_trial_15", 
    #      mocks_name = "kids_1000_cosmology", 
    #      sim_number = 4000,
    #      compressed_name = 'compressed_data', 
    #      linear_compression = True)   
    
    # main(deriv_params = ['cosmological_parameters--omega_m', 'cosmological_parameters--sigma_8',
    #                      'intrinsic_alignment_parameters--a', 'cosmological_parameters--n_s', 
    #                      'halo_model_parameters--a'], 
    #      data_params = ['theory'], 
    #      theta_names = ['cosmological_parameters--omega_m', 'cosmological_parameters--sigma_8',
    #                      'intrinsic_alignment_parameters--a', 'cosmological_parameters--n_s', 
    #                      'halo_model_parameters--a'], 
    #      mocks_dir = "/mnt/Node-Temp/cosmology/kcap_output/kids_1000_mocks_trial_17", 
    #      mocks_name = "kids_1000_cosmology", 
    #      sim_number = 4000,
    #      compressed_name = 'compressed_data_new_derivs', 
    #      linear_compression = True)  
    
    main(deriv_params = ['intrinsic_alignment_parameters--a'], 
         data_params = ['theory'], 
         theta_names = ['intrinsic_alignment_parameters--a'], 
         mocks_dir = "/mnt/Node-Temp/cosmology/kcap_output/deriv_test/varying_aia", 
         mocks_name = "kids_1000_cosmology", 
         sim_number = 1200,
         compressed_name = 'a_ia', 
         linear_compression = True,
         noisey_data = False) 
    main(deriv_params = ['halo_model_parameters--a'], 
         data_params = ['theory'], 
         theta_names = ['halo_model_parameters--a'], 
         mocks_dir = "/mnt/Node-Temp/cosmology/kcap_output/deriv_test/varying_haloa", 
         mocks_name = "kids_1000_cosmology", 
         sim_number = 1200,
         compressed_name = 'halo_a', 
         linear_compression = True,
         noisey_data = False) 
    main(deriv_params = ['cosmological_parameters--n_s'], 
         data_params = ['theory'], 
         theta_names = ['cosmological_parameters--n_s'], 
         mocks_dir = "/mnt/Node-Temp/cosmology/kcap_output/deriv_test/varying_ns", 
         mocks_name = "kids_1000_cosmology", 
         sim_number = 1200,
         compressed_name = 'n_s', 
         linear_compression = True,
         noisey_data = False) 
    main(deriv_params = ['cosmological_parameters--omch2'], 
         data_params = ['theory'], 
         theta_names = ['cosmological_parameters--omch2'], 
         mocks_dir = "/mnt/Node-Temp/cosmology/kcap_output/deriv_test/varying_omch2", 
         mocks_name = "kids_1000_cosmology", 
         sim_number = 1200,
         compressed_name = 'omch2', 
         linear_compression = True,
         noisey_data = False) 
    main(deriv_params = ['cosmological_parameters--ombh2'], 
         data_params = ['theory'], 
         theta_names = ['cosmological_parameters--ombh2'], 
         mocks_dir = "/mnt/Node-Temp/cosmology/kcap_output/deriv_test/varying_ombh2", 
         mocks_name = "kids_1000_cosmology", 
         sim_number = 1200,
         compressed_name = 'ombh2', 
         linear_compression = True,
         noisey_data = False) 
    main(deriv_params = ['cosmological_parameters--sigma_8'], 
         data_params = ['theory'], 
         theta_names = ['cosmological_parameters--sigma_8'], 
         mocks_dir = "/mnt/Node-Temp/cosmology/kcap_output/deriv_test/varying_sigma_8", 
         mocks_name = "kids_1000_cosmology", 
         sim_number = 1200,
         compressed_name = 'sigma_8', 
         linear_compression = True,
         noisey_data = False) 
    
    # main(deriv_params = ['cosmological_parameters--omega_m', 'cosmological_parameters--sigma_8',
    #                      'intrinsic_alignment_parameters--a', 'cosmological_parameters--n_s', 
    #                      'cosmological_parameters--h0', 'halo_model_parameters--a'], 
    #      data_params = ['theory'], 
    #      theta_names = ['cosmological_parameters--omega_m', 'cosmological_parameters--sigma_8',
    #                      'intrinsic_alignment_parameters--a', 'cosmological_parameters--n_s', 
    #                      'cosmological_parameters--h0', 'halo_model_parameters--a'], 
    #      mocks_dir = "/mnt/Node-Data/cosmology/kcap_output/kids_1000_mocks_trial_15", 
    #      mocks_name = "kids_1000_cosmology", 
    #      sim_number = 4000,
    #      compressed_name = 'compressed_data_binned', 
    #      linear_compression = True)   

    # data_params = ['shear_xi_plus_binned', 'shear_xi_minus_binned']
    # mocks_dir = '/home/ruyi/cosmology/kcap_output/kids_deriv_test'
    # mocks_name = 'domega_m_fixed_s_8'
    # mocks_name = 'domega_m_fixed_sigma_8'
    # mocks_name = 'dsigma_8_fixed_omega_m'
    # mocks_name = 'ds_8_fixed_omega_m'
    
    # data_vector = get_fiducial_deriv(fiducial_run = 0, deriv_params = ['s_8'], data_params = ['shear_xi_plus_binned', 'shear_xi_minus_binned'], mocks_dir = mocks_dir, mocks_name = mocks_name, bin_order = None)
    # data_vector = data_vector.flatten()
    # data_vector_dict = kcap_methods.get_thetas(mock_run = 0, vals_to_read = data_params, mocks_dir = mocks_dir, mocks_name = mocks_name) # The datavector stored as a dict of 2 flattened numpy arrays
    # data_vector = np.array([])
    # for data_param in data_params:
    #     data_vector = np.append(data_vector, data_vector_dict[data_param])
    
    # write_file(input_array = data_vector, file_location = mocks_dir + '/', file_name = 'thetas')