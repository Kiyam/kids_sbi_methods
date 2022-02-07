import numpy as np
import os
import errno
import kcap_methods

#TODO Still need to figure out how to read the covariance matrix from kcap

class delfi_score_compress:
    def __init__(self, theta_fiducial, fid_vector, deriv_matrix, inv_covariance, fisher_matrix, nuisance_parameter_indices = None):
        self.theta_fiducial = theta_fiducial
        self.fid_vector = fid_vector
        self.deriv_matrix = deriv_matrix
        self.inv_covariance = inv_covariance
        self.fisher_matrix = fisher_matrix
        
        if nuisance_parameter_indices is not None:
            self.nuisance_parameter_indices = nuisance_parameter_indices
        
    def score_compress(self, data_vector):
        """
        General Score compression
        """
        data_diff = data_vector - self.fid_vector  
        linear_score = np.dot(self.deriv_matrix, np.dot(self.inv_covariance, data_diff))
        inverse_fisher = np.linalg.inv(self.fisher_matrix)
        if self.nuisance_parameter_indices:
            # for this to work, the parameters to be marginalised out have to be at the end of the matrix
            full_mle_fiducial = self.theta_fiducial + np.transpose(np.dot(inverse_fisher, np.transpose(linear_score)))
            
            mle_fiducial_wanted = full_mle_fiducial[: min(self.nuisance_parameter_indices)]
            mle_fiducial_nuisance = full_mle_fiducial[min(self.nuisance_parameter_indices) :]
            sub_nuisance_fisher = self.fisher_matrix[:, self.nuisance_parameter_indices]
            nuisance_inv_fisher = np.linalg.inv(self.fisher_matrix[np.ix_(self.nuisance_parameter_indices, self.nuisance_parameter_indices)])
            
            mle_fiducial = mle_fiducial_wanted - np.dot(sub_nuisance_fisher, np.dot(nuisance_inv_fisher, mle_fiducial_nuisance))
        else:
            mle_fiducial = self.theta_fiducial + np.transpose(np.dot(inverse_fisher, np.transpose(linear_score)))
        return mle_fiducial 

def nuisance_hardened_mle(fisher, score, nuisance_index = [-1]):  
    wanted_score = score[:,:min(nuisance_index)]
    nuisance_score = score[:,min(nuisance_index):]
    sub_nuisance_fisher = fisher[np.ix_(list(range(0, min(nuisance_index))), nuisance_index)]
    
    if len(nuisance_index) == 1:
        nuisance_inv_fisher = 1/fisher[nuisance_index[0]][nuisance_index[0]]
        marginalised_component = np.dot(sub_nuisance_fisher, (nuisance_inv_fisher * nuisance_score).T).T
    else:
        nuisance_inv_fisher = np.linalg.inv(fisher[np.ix_(nuisance_index, nuisance_index)])
        marginalised_component = np.dot(sub_nuisance_fisher, np.dot(nuisance_inv_fisher, nuisance_score.T)).T
    
    marginalised_score = wanted_score - marginalised_component
    marginalised_inv_fisher = np.linalg.inv(fisher)[:min(nuisance_index), :min(nuisance_index)]
    
    return marginalised_score, marginalised_inv_fisher

def calc_cholesky(covariance, array, method = "corr"):
    L = np.linalg.cholesky(covariance) 
    if method == "corr":
        return np.dot(L, array)
    else:
        inv_L = np.linalg.inv(L)
        return np.dot(inv_L, array))

def calc_fisher(inv_covariance, deriv_matrix):
    """
    Computes the fisher matrix based on the params set, 
    e.g params = ["shear_xi_plus_omch2_deriv", "shear_xi_plus_sigma_8_deriv", "shear_xi_minus_omch2_deriv", "shear_xi_minus_sigma_8_deriv"]
    """

    fisher_matrix = np.dot(deriv_matrix, np.dot(inv_covariance, deriv_matrix.T))

    return fisher_matrix

def calc_gaussian_posterior(data_vector, fid_vector, inv_covariance):
    """
    Calculates the regular posterior assuming Gaussian likelihood.
    """

    data_diff = data_vector - fid_vector
    gaus_post = np.exp(-0.5 * np.einsum('ij, ij -> i', data_diff, np.dot(inv_covariance, data_diff.T).T))
    
    return gaus_post

def calc_fisher_likelihood(theta, fid_theta, fisher):
    """
    Calculates the theta posterior assuming Gaussian likelihood using the fisher matrix.
    """

    theta_diff = theta - fid_theta
    fisher_likelihood = np.exp(-0.5 * np.dot(theta_diff, np.dot(fisher, theta_diff.T).T))   
    
    return fisher_likelihood     

def calc_score_compress(data_vector, fid_vector, deriv_matrix, inv_covariance, cov_deriv_tensor = None):
    """
    General Score compression
    """
    data_diff = data_vector - fid_vector  
    # Now to do the matrix multiplications for score compression
    linear_score = np.dot(deriv_matrix, np.dot(inv_covariance, data_diff.T)).T
    if cov_deriv_tensor is None:
        print("Linear Score Compression Done!")
        return linear_score
    else:
        #NOTE This non linear part is untested
        # cov_score = np.dot(np.dot(np.transpose(data_diff), inv_covariance), np.transpose(np.dot(cov_deriv_tensor, np.dot(inv_covariance, data_diff))))
        cov_score = np.dot(np.dot(data_diff.T, inv_covariance), np.dot(cov_deriv_tensor, np.dot(inv_covariance, data_diff.T)).T )              
        score = linear_score + cov_score
        print("Non-Linear Score Compression Done")
        return score
    
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

def main(deriv_params, data_params, theta_names, mocks_dir, mocks_name, sim_number, linear = True,
         fiducial_mocks_dir = '/share/data1/klin/kcap_out/kids_fiducial_data_mocks', fiducial_mocks_name = 'kids_1000_cosmology_fiducial', fiducial_run = 0,
         data_mocks_dir = '/share/data1/klin/kcap_out/kids_fiducial_data_mocks', data_mocks_name = 'kids_1000_cosmology_noiseless', data_run = 0,
         compressed_name = 'compressed_data', cov_inv_method = 'eigen', noisey_data = True):

    inv_covariance = kcap_methods.get_inv_covariance(mock_run = fiducial_run, which_cov = cov_inv_method, mocks_dir = fiducial_mocks_dir, mocks_name = fiducial_mocks_name)
    deriv_matrix = kcap_methods.get_fiducial_deriv(fiducial_run = fiducial_run, deriv_params = deriv_params, data_params = data_params, mocks_dir = fiducial_mocks_dir, mocks_name = fiducial_mocks_name)
    fid_vector = kcap_methods.get_single_data_vector(mock_run = fiducial_run, data_params = data_params, mocks_dir = fiducial_mocks_dir, mocks_name = fiducial_mocks_name)

    data_vector = kcap_methods.get_single_data_vector(mock_run = data_run, data_params = data_params, mocks_dir = data_mocks_dir, mocks_name = data_mocks_name)
    data_theta = np.array(list(kcap_methods.get_params(mock_run = data_run, vals_to_read = theta_names, mocks_dir = mocks_dir, mocks_name = mocks_name).values()))

    sim_thetas = kcap_methods.get_sim_batch_thetas(sim_number = sim_number, theta_names = theta_names, mocks_dir = mocks_dir, mocks_name = mocks_name)
    sim_data_vectors = kcap_methods.get_sim_batch_data_vectors(sim_number, data_vector_length = 270, mocks_dir = mocks_dir, mocks_name = mocks_name, noisey_data = noisey_data)
    
    sim_thetas = np.vstack([sim_thetas, data_theta])
    sim_data_vectors = np.vstack([sim_data_vectors, data_vector])

    if linear is not True:
        cov_deriv_tensor = kcap_methods.get_fiducial_cov_deriv(fiducial_run = fiducial_run, deriv_matrix = deriv_matrix, deriv_params = deriv_params, mocks_dir = fiducial_mocks_dir, mocks_name = fiducial_mocks_name)
    else:
        cov_deriv_tensor = None

    file_loc = mocks_dir + '/' + compressed_name

    compressed_data = calc_score_compress(data_vector = sim_data_vectors, fid_vector = fid_vector, deriv_matrix = deriv_matrix, inv_covariance = inv_covariance, cov_deriv_tensor = cov_deriv_tensor)
    fisher_matrix = calc_fisher(inv_covariance = inv_covariance , deriv_matrix = deriv_matrix)
    posterior = calc_gaussian_posterior(data_vector = sim_data_vectors, fid_vector = fid_vector, inv_covariance = inv_covariance)
    likelihood = kcap_methods.get_sim_batch_likelihood(sim_number = sim_number, mocks_dir = mocks_dir, mocks_name = mocks_name)
    data_likelihood = kcap_methods.get_likelihood(mock_run = data_run, mocks_dir = data_mocks_dir, mocks_name = data_mocks_name)
    likelihood = np.append(likelihood, data_likelihood)

    write_file(input_array = sim_thetas, file_location = file_loc, file_name = 'thetas')
    write_file(input_array = compressed_data, file_location = file_loc, file_name = 'compressed_score')
    write_file(input_array = compressed_data[-1], file_location = file_loc, file_name = 'score_compressed_data')
    write_file(input_array = fisher_matrix, file_location = file_loc, file_name = 'fisher_matrix')
    write_file(input_array = posterior, file_location = file_loc, file_name = 'posterior')
    write_file(input_array = likelihood, file_location = file_loc, file_name = 'likelihood')

if __name__ == "__main__":    
    # ['\sigma_8', '\omega_c', '\omega_b', 'A_{IA}', 'n_s', 'a_{halo}']
    # fisher = compute_fisher(fiducial_run = 0, 
    #                         deriv_params = ['cosmological_parameters--sigma_8', 
    #                                         'cosmological_parameters--omega_m',
    #                                         'intrinsic_alignment_parameters--a',
    #                                         'cosmological_parameters--n_s',
    #                                         'halo_model_parameters--a',
    #                                         'cosmological_parameters--ombh2',
    #                                         'nofz_shifts--bias_1',
    #                                         'nofz_shifts--bias_2',
    #                                         'nofz_shifts--bias_3',
    #                                         'nofz_shifts--bias_4',
    #                                         'nofz_shifts--bias_5'], 
    #                         data_params = ['theory'], 
    #                         mocks_dir = '/home/ruyi/cosmology/kcap_output/kids_mocks', 
    #                         mocks_name = 'kids_1000_cosmology_noiseless')
    
    # write_file(input_array = fisher, file_location = '/mnt/Node-Temp/cosmology/kcap_output/kids_1000_mocks_trial_28/fisher', 
    #            file_name = 'fisher_matrix_omega_m') 
    
    # main(deriv_params = ['cosmological_parameters--h0'], 
    #     data_params = ['theory'], 
    #     theta_names = ['cosmological_parameters--h0'], 
    #     mocks_dir = "/mnt/Node-Temp/cosmology/kids_deriv_tests/fisher_1d/varying_h0", 
    #     mocks_name = "kids_1000_cosmology", 
    #     sim_number = 800,
    #     compressed_name = 'h0', 
    #     cov_inv_method = 'eigen',
    #     linear_compression = True,
    #     noisey_data = True)
    
    main(deriv_params = ['cosmological_parameters--sigma_8', 
                         'cosmological_parameters--omch2',
                         'intrinsic_alignment_parameters--a',
                         'cosmological_parameters--n_s',
                         'halo_model_parameters--a',
                         'cosmological_parameters--h0',
                         'cosmological_parameters--ombh2',
                         'nofz_shifts--bias_1',
                         'nofz_shifts--bias_2',
                         'nofz_shifts--bias_3',
                         'nofz_shifts--bias_4',
                         'nofz_shifts--bias_5'], 
         data_params = ['theory'], 
         theta_names = ['cosmological_parameters--sigma_8', 
                        'cosmological_parameters--omch2',
                        'intrinsic_alignment_parameters--a',
                        'cosmological_parameters--n_s',
                        'halo_model_parameters--a',
                        'cosmological_parameters--h0',
                        'cosmological_parameters--ombh2',
                        'nofz_shifts--bias_1',
                        'nofz_shifts--bias_2',
                        'nofz_shifts--bias_3',
                        'nofz_shifts--bias_4',
                        'nofz_shifts--bias_5'], 
         mocks_dir = "/share/data1/klin/kcap_out/kids_1000_mocks/trial_31/hypercube", 
         mocks_name = "kids_1000_cosmology_with_nz_shifts_corr", 
         sim_number = 5,
         compressed_name = 'a_test', 
         cov_inv_method = 'eigen',
         noisey_data = True)