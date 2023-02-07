import numpy as np
import kcap_methods
import glob
from pathlib import Path

class delfi_score_compress:
    def __init__(self, theta_fiducial, fid_vector, deriv_matrix, inv_covariance, fisher_matrix, nuisance_parameter_indices = None):
        self.theta_fiducial = theta_fiducial
        self.fid_vector = fid_vector
        self.deriv_matrix = deriv_matrix
        self.inv_covariance = inv_covariance
        self.fisher_matrix = fisher_matrix
        self.inverse_fisher = np.linalg.inv(fisher_matrix)
        self.nuisance_parameter_indices = nuisance_parameter_indices
        
    def score_compress(self, data_vector, thetas):
        """
        General Score compression
        """
        data_diff = data_vector - self.fid_vector  
        linear_score = np.dot(self.deriv_matrix, np.dot(self.inv_covariance, data_diff.T)).T
        
        if self.nuisance_parameter_indices is not None:
            if len(data_diff.shape) == 1:
                wanted_score = linear_score[:min(self.nuisance_parameter_indices)]
                nuisance_score = linear_score[min(self.nuisance_parameter_indices):]
                thetas = thetas[:min(self.nuisance_parameter_indices)]
            else:
                wanted_score = linear_score[:,:min(self.nuisance_parameter_indices)]
                nuisance_score = linear_score[:,min(self.nuisance_parameter_indices):]
                thetas = thetas[:,:min(self.nuisance_parameter_indices)]
                
            sub_nuisance_fisher = self.fisher_matrix[np.ix_(list(range(0, min(self.nuisance_parameter_indices))), self.nuisance_parameter_indices)]
            
            if len(self.nuisance_parameter_indices) == 1:
                nuisance_inv_fisher = 1/self.fisher_matrix[self.nuisance_parameter_indices[0]][self.nuisance_parameter_indices[0]]
                marginalised_component = np.dot(sub_nuisance_fisher, (nuisance_inv_fisher * nuisance_score).T).T
            else:
                nuisance_inv_fisher = np.linalg.inv(self.fisher_matrix[np.ix_(self.nuisance_parameter_indices, self.nuisance_parameter_indices)])
                marginalised_component = np.dot(sub_nuisance_fisher, np.dot(nuisance_inv_fisher, nuisance_score.T)).T
            
            marginalised_score = wanted_score - marginalised_component
            marginalised_inv_fisher = np.linalg.inv(self.fisher_matrix)[:min(self.nuisance_parameter_indices), :min(self.nuisance_parameter_indices)]
            
            mle_fiducial = self.theta_fiducial[:min(self.nuisance_parameter_indices)] + np.dot(marginalised_inv_fisher, marginalised_score.T).T
        else:
            mle_fiducial = self.theta_fiducial + np.dot(self.inverse_fisher, linear_score.T).T

        return mle_fiducial, thetas

def nuisance_hardened_score(fisher, score, nuisance_index = [-1]):  
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
        return np.dot(inv_L, array)

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
    Path(file_location).mkdir(parents=True, exist_ok=True)
    np.savetxt(outfile, input_array)
    print("File succesfully saved as %s" % str(outfile))

def main(deriv_params, data_params, deriv_data_params, theta_names, mocks_dir, mocks_name, sim_number, linear = True,
         inv_covariance = '/share/data1/klin/kcap_out/kids_fiducial_data_mocks/covariances/inv_covariance.txt',
         fiducial_mocks_dir = '/share/data1/klin/kcap_out/kids_fiducial_data_mocks', fiducial_mocks_name = 'kids_1000_cosmology_fiducial', fiducial_run = None,
         data_mocks_dir = '/share/data1/klin/kcap_out/kids_fiducial_data_mocks', data_mocks_name = 'kids_1000_cosmology_data', data_run = None,
         compressed_file_loc = None, compressed_name = 'compressed_data'):
    
    inv_covariance = np.loadtxt(inv_covariance)
    deriv_matrix = kcap_methods.get_fiducial_deriv(deriv_params = deriv_params, data_params = deriv_data_params, fiducial_run = None, mocks_dir = fiducial_mocks_dir, mocks_name = fiducial_mocks_name)

    fid_tar = kcap_methods.read_kcap_values(mocks_dir = fiducial_mocks_dir, mocks_name = fiducial_mocks_name)
    fid_vector = fid_tar.read_vals(vals_to_read = deriv_data_params, output_type = 'as_flat')
    fid_tar.close()
    
    data_tar = kcap_methods.read_kcap_values(mocks_dir = data_mocks_dir, mocks_name = data_mocks_name)
    data_vector = data_tar.read_vals(vals_to_read = deriv_data_params, output_type = 'as_flat')
    data_theta = data_tar.read_params(parameter_list = theta_names, output_type = 'as_flat')
    data_tar.close()
    
    sim_data_vectors, sim_thetas = kcap_methods.get_sim_batch_data_params(sim_number = sim_number, param_names = theta_names, data_names = data_params, mocks_dir = mocks_dir, mocks_name = mocks_name)
    
    sim_thetas = np.vstack([sim_thetas, data_theta])
    sim_data_vectors = np.vstack([sim_data_vectors, data_vector])

    if linear is not True:
        cov_deriv_tensor = kcap_methods.get_fiducial_cov_deriv(fiducial_run = fiducial_run, deriv_matrix = deriv_matrix, deriv_params = deriv_params, mocks_dir = fiducial_mocks_dir, mocks_name = fiducial_mocks_name)
    else:
        cov_deriv_tensor = None

    file_loc = compressed_file_loc + '/' + compressed_name

    compressed_data = calc_score_compress(data_vector = sim_data_vectors, fid_vector = fid_vector, deriv_matrix = deriv_matrix, inv_covariance = inv_covariance, cov_deriv_tensor = cov_deriv_tensor)
    fisher_matrix = calc_fisher(inv_covariance = inv_covariance , deriv_matrix = deriv_matrix)

    write_file(input_array = sim_thetas, file_location = file_loc, file_name = 'thetas')
    write_file(input_array = compressed_data, file_location = file_loc, file_name = 'compressed_score')
    write_file(input_array = compressed_data[-1], file_location = file_loc, file_name = 'score_compressed_data')
    write_file(input_array = fisher_matrix, file_location = file_loc, file_name = 'fisher_matrix')
    
def main_without_fetch(deriv_params, data_params, deriv_data_params, mock_data_params, theta_names, theta_file, data_vector_file, covariance, linear = True,
         fiducial_mocks_dir = '/share/data1/klin/kcap_out/kids_fiducial_data_mocks', fiducial_mocks_name = 'kids_1000_cosmology_fiducial', fiducial_run = 0,
         data_mocks_dir = '/share/data1/klin/kcap_out/kids_fiducial_data_mocks', data_mocks_name = 'kids_1000_cosmology_data', data_run = 0,
         compressed_file_loc = None, compressed_name = 'compressed_data'):

    inv_covariance = kcap_methods.calc_inv_covariance(covariance)
    
    deriv_matrix = kcap_methods.get_fiducial_deriv(fiducial_run = fiducial_run, deriv_params = deriv_params, data_params = deriv_data_params, mocks_dir = fiducial_mocks_dir, mocks_name = fiducial_mocks_name)
    fid_vector = kcap_methods.get_single_data_vector(mock_run = fiducial_run, data_params = deriv_data_params, mocks_dir = fiducial_mocks_dir, mocks_name = fiducial_mocks_name)

    data_vector = kcap_methods.get_single_data_vector(mock_run = data_run, data_params = mock_data_params, mocks_dir = data_mocks_dir, mocks_name = data_mocks_name)
    data_theta = np.array(list(kcap_methods.get_params(mock_run = data_run, vals_to_read = theta_names, mocks_dir = data_mocks_dir, mocks_name = data_mocks_name).values()))

    sim_thetas = np.genfromtxt(theta_file)
    sim_data_vectors = np.genfromtxt(data_vector_file)
        
    sim_thetas = np.vstack([sim_thetas, data_theta])
    sim_data_vectors = np.vstack([sim_data_vectors, data_vector])

    if linear is not True:
        cov_deriv_tensor = kcap_methods.get_fiducial_cov_deriv(fiducial_run = fiducial_run, deriv_matrix = deriv_matrix, deriv_params = deriv_params, mocks_dir = fiducial_mocks_dir, mocks_name = fiducial_mocks_name)
    else:
        cov_deriv_tensor = None

    file_loc = compressed_file_loc + '/' + compressed_name

    compressed_data = calc_score_compress(data_vector = sim_data_vectors, fid_vector = fid_vector, deriv_matrix = deriv_matrix, inv_covariance = inv_covariance, cov_deriv_tensor = cov_deriv_tensor)
    fisher_matrix = calc_fisher(inv_covariance = inv_covariance , deriv_matrix = deriv_matrix)

    write_file(input_array = sim_thetas, file_location = file_loc, file_name = 'thetas')
    write_file(input_array = compressed_data, file_location = file_loc, file_name = 'compressed_score')
    write_file(input_array = compressed_data[-1], file_location = file_loc, file_name = 'score_compressed_data')
    write_file(input_array = fisher_matrix, file_location = file_loc, file_name = 'fisher_matrix')

def calculate_fisher(data_params, deriv_params, fiducial_mocks_dir, fiducial_mocks_name, fiducial_run, file_loc3, file_name = 'fisher_matrix', which_cov = "theory_data_covariance--covariance"):
    inv_covariance = kcap_methods.get_inv_covariance(mock_run = fiducial_run, which_cov = which_cov, mocks_dir = fiducial_mocks_dir, mocks_name = fiducial_mocks_name)
    deriv_matrix = kcap_methods.get_fiducial_deriv(fiducial_run = fiducial_run, deriv_params = deriv_params, data_params = data_params, mocks_dir = fiducial_mocks_dir, mocks_name = fiducial_mocks_name)
    fisher_matrix = calc_fisher(inv_covariance = inv_covariance , deriv_matrix = deriv_matrix)
    write_file(input_array = fisher_matrix, file_location = file_loc, file_name = file_name)

def cov_varied_check(deriv_params, data_params, deriv_data_params, theta_names, mocks_dir, mocks_name, sim_number, covariance_file_paths = "/share/data1/klin/kcap_out/kids_fiducial_data_mocks/varied_covariances/*", 
                     fiducial_mocks_dir = '/share/data1/klin/kcap_out/kids_fiducial_data_mocks', fiducial_mocks_name = 'kids_1000_cosmology_fiducial', fiducial_run = 0,
                     data_mocks_dir = '/share/data1/klin/kcap_out/kids_fiducial_data_mocks', data_mocks_name = 'kids_1000_cosmology_data', data_run = 0, 
                     compressed_file_loc = None, compressed_name = 'compressed_data', cov_inv_method = 'eigen'):

    covariance_files = glob.glob(covariance_file_paths)

    deriv_matrix = kcap_methods.get_fiducial_deriv(fiducial_run = fiducial_run, deriv_params = deriv_params, data_params = deriv_data_params, mocks_dir = fiducial_mocks_dir, mocks_name = fiducial_mocks_name)
    fid_vector = kcap_methods.get_single_data_vector(mock_run = fiducial_run, data_params = deriv_data_params, mocks_dir = fiducial_mocks_dir, mocks_name = fiducial_mocks_name)

    data_vector = kcap_methods.get_single_data_vector(mock_run = data_run, data_params = data_params, mocks_dir = data_mocks_dir, mocks_name = data_mocks_name)
    data_theta = np.array(list(kcap_methods.get_params(mock_run = data_run, vals_to_read = theta_names, mocks_dir = data_mocks_dir, mocks_name = data_mocks_name).values()))

    sim_thetas = kcap_methods.get_sim_batch_thetas(sim_number = sim_number, theta_names = theta_names, mocks_dir = mocks_dir, mocks_name = mocks_name)
    sim_data_vectors = kcap_methods.get_sim_batch_data_vectors(sim_number, data_params = data_params, data_vector_length = len(data_vector), mocks_dir = mocks_dir, mocks_name = mocks_name)
    
    for file in covariance_files:
        covariance = np.genfromtxt(file)
        inv_covariance = kcap_methods.calc_inv_covariance(covariance = covariance, which_cov = cov_inv_method)

        current_sim_thetas = np.vstack([sim_thetas, data_theta])
        current_sim_data_vectors = np.vstack([sim_data_vectors, data_vector])

        compressed_data = calc_score_compress(data_vector = current_sim_data_vectors, fid_vector = fid_vector, deriv_matrix = deriv_matrix, inv_covariance = inv_covariance, cov_deriv_tensor = None)
        fisher_matrix = calc_fisher(inv_covariance = inv_covariance , deriv_matrix = deriv_matrix)
        posterior = calc_gaussian_posterior(data_vector = current_sim_data_vectors, fid_vector = fid_vector, inv_covariance = inv_covariance)
        likelihood = kcap_methods.get_sim_batch_likelihood(sim_number = sim_number, mocks_dir = mocks_dir, mocks_name = mocks_name)
        data_likelihood = kcap_methods.get_likelihood(mock_run = 0, mocks_dir = data_mocks_dir, mocks_name = data_mocks_name)
        likelihood = np.append(likelihood, data_likelihood)

        save_name = file.split(sep = "/")[-1][:-4]
        
        file_loc = compressed_file_loc + '/' + compressed_name + '_' + save_name

        write_file(input_array = current_sim_thetas, file_location = file_loc, file_name = 'thetas')
        write_file(input_array = compressed_data, file_location = file_loc, file_name = 'compressed_score')
        write_file(input_array = compressed_data[-1], file_location = file_loc, file_name = 'score_compressed_data')
        write_file(input_array = fisher_matrix, file_location = file_loc, file_name = 'fisher_matrix')