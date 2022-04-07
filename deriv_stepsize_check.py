import numpy as np
import sys
sys.path.append('/share/splinter/klin/kcap_methods/')
import kcap_methods
import shutil

def run_varying_stepsize(step_sizes, stencil_pts, mocks_dir = None, mocks_name = None, deriv_ini_file = None, deriv_values_file = None, deriv_values_list_file = None, base_folder = None):
    for step_size in step_sizes:
        shutil.copytree(base_folder, mocks_dir+'/'+mocks_name+'_'+str(step_size))

        kcap_methods.run_kcap_deriv(mock_run = step_size, 
                    param_to_vary = 'cosmological_parameters--omch2', 
                    params_to_fix = ['cosmological_parameters--ombh2', 'cosmological_parameters--sigma_8', 
                                     'cosmological_parameters--n_s', 'halo_model_parameters--a', 
                                     'intrinsic_alignment_parameters--a', 'cosmological_parameters--h0'],
                    vals_to_diff = ["bandpowers--theory_bandpower_cls"],
                    step_size = float(step_size),
                    stencil_pts = stencil_pts,
                    mocks_dir = mocks_dir, 
                    mocks_name = mocks_name,
                    cleanup = 1,
                    deriv_ini_file = deriv_ini_file, 
                    deriv_values_file = deriv_values_file, 
                    deriv_values_list_file = deriv_values_list_file)
        kcap_methods.run_kcap_deriv(mock_run = step_size, 
                    param_to_vary = 'cosmological_parameters--ombh2', 
                    params_to_fix = ['cosmological_parameters--omch2', 'cosmological_parameters--sigma_8', 
                                     'cosmological_parameters--n_s', 'halo_model_parameters--a', 
                                     'intrinsic_alignment_parameters--a', 'cosmological_parameters--h0'],
                    vals_to_diff = ["bandpowers--theory_bandpower_cls"],
                    step_size = float(step_size),
                    stencil_pts = stencil_pts,
                    mocks_dir = mocks_dir, 
                    mocks_name = mocks_name,
                    cleanup = 1,
                    deriv_ini_file = deriv_ini_file, 
                    deriv_values_file = deriv_values_file, 
                    deriv_values_list_file = deriv_values_list_file)
        kcap_methods.run_kcap_deriv(mock_run = step_size, 
                    param_to_vary = 'cosmological_parameters--sigma_8', 
                    params_to_fix = ['cosmological_parameters--ombh2', 'cosmological_parameters--omch2', 
                                     'cosmological_parameters--n_s', 'halo_model_parameters--a', 
                                     'intrinsic_alignment_parameters--a', 'cosmological_parameters--h0'],
                    vals_to_diff = ["bandpowers--theory_bandpower_cls"],
                    step_size = float(step_size),
                    stencil_pts = stencil_pts,
                    mocks_dir = mocks_dir, 
                    mocks_name = mocks_name,
                    cleanup = 1,
                    deriv_ini_file = deriv_ini_file, 
                    deriv_values_file = deriv_values_file, 
                    deriv_values_list_file = deriv_values_list_file)
        kcap_methods.run_kcap_deriv(mock_run = step_size, 
                    param_to_vary = 'cosmological_parameters--n_s', 
                    params_to_fix = ['cosmological_parameters--ombh2', 'cosmological_parameters--sigma_8', 
                                     'cosmological_parameters--omch2', 'halo_model_parameters--a', 
                                     'intrinsic_alignment_parameters--a', 'cosmological_parameters--h0'],
                    vals_to_diff = ["bandpowers--theory_bandpower_cls"],
                    step_size = float(step_size),
                    stencil_pts = stencil_pts,
                    mocks_dir = mocks_dir, 
                    mocks_name = mocks_name,
                    cleanup = 1,
                    deriv_ini_file = deriv_ini_file, 
                    deriv_values_file = deriv_values_file, 
                    deriv_values_list_file = deriv_values_list_file)
        kcap_methods.run_kcap_deriv(mock_run = step_size, 
                    param_to_vary = 'halo_model_parameters--a', 
                    params_to_fix = ['cosmological_parameters--ombh2', 'cosmological_parameters--sigma_8', 
                                     'cosmological_parameters--n_s', 'cosmological_parameters--omch2', 
                                     'intrinsic_alignment_parameters--a', 'cosmological_parameters--h0'],
                    vals_to_diff = ["bandpowers--theory_bandpower_cls"],
                    step_size = float(step_size),
                    stencil_pts = stencil_pts,
                    mocks_dir = mocks_dir, 
                    mocks_name = mocks_name,
                    cleanup = 1,
                    deriv_ini_file = deriv_ini_file, 
                    deriv_values_file = deriv_values_file, 
                    deriv_values_list_file = deriv_values_list_file)
        kcap_methods.run_kcap_deriv(mock_run = step_size, 
                    param_to_vary = 'intrinsic_alignment_parameters--a', 
                    params_to_fix = ['cosmological_parameters--ombh2', 'cosmological_parameters--sigma_8', 
                                     'cosmological_parameters--n_s', 'halo_model_parameters--a', 
                                     'cosmological_parameters--omch2', 'cosmological_parameters--h0'],
                    vals_to_diff = ["bandpowers--theory_bandpower_cls"],
                    step_size = float(step_size),
                    stencil_pts = stencil_pts,
                    mocks_dir = mocks_dir, 
                    mocks_name = mocks_name,
                    cleanup = 1,
                    deriv_ini_file = deriv_ini_file, 
                    deriv_values_file = deriv_values_file, 
                    deriv_values_list_file = deriv_values_list_file)
        kcap_methods.run_kcap_deriv(mock_run = step_size, 
                    param_to_vary = 'cosmological_parameters--h0', 
                    params_to_fix = ['cosmological_parameters--ombh2', 'cosmological_parameters--sigma_8', 
                                     'cosmological_parameters--n_s', 'halo_model_parameters--a', 
                                     'intrinsic_alignment_parameters--a', 'cosmological_parameters--omch2'],
                    vals_to_diff = ["bandpowers--theory_bandpower_cls"],
                    step_size = float(step_size),
                    stencil_pts = stencil_pts,
                    mocks_dir = mocks_dir, 
                    mocks_name = mocks_name,
                    cleanup = 1,
                    deriv_ini_file = deriv_ini_file, 
                    deriv_values_file = deriv_values_file, 
                    deriv_values_list_file = deriv_values_list_file)

def compute_stepsize_fisher(stepsizes, deriv_params, data_params, inv_covariance, mocks_dir = None, mocks_name = None, bin_order = None):
    """
    Computes the fisher matrix based on the params set, 
    e.g params = ["shear_xi_plus_omch2_deriv", "shear_xi_plus_sigma_8_deriv", "shear_xi_minus_omch2_deriv", "shear_xi_minus_sigma_8_deriv"]
    """    

    deriv_matrix = np.zeros(shape = (len(deriv_params), inv_covariance.shape[0]))
    for i, deriv_param in enumerate(deriv_params):
        deriv_vals_to_get = [data_param.split(sep="--")[0]+'_'+deriv_param+'_deriv--'+data_param.split(sep="--")[1] for data_param in data_params]
        deriv_vector_dict = kcap_methods.get_values(mock_run = stepsizes[i], vals_to_read = deriv_vals_to_get, mocks_dir = mocks_dir, mocks_name = mocks_name, bin_order = bin_order)
        deriv_vector = np.array([])
        for data_param in data_params:
            deriv_vector = np.append(deriv_vector, deriv_vector_dict[data_param.split(sep="--")[0]+'_'+deriv_param+'_deriv--'+data_param.split(sep="--")[1]])
        
        deriv_matrix[i] += deriv_vector

    fisher_matrix = np.dot(deriv_matrix, np.dot(inv_covariance, np.transpose(deriv_matrix)))
    return fisher_matrix

def calculate_fom(param_index, full_fisher_matrix):
    """
    Computes the fisher matrix based on the params set, 
    e.g params = ["shear_xi_plus_omch2_deriv", "shear_xi_plus_sigma_8_deriv", "shear_xi_minus_omch2_deriv", "shear_xi_minus_sigma_8_deriv"]
    """
    reduced_cov = np.linalg.inv(full_fisher_matrix)[param_index][param_index]
    fom = np.sqrt(1/reduced_cov)
    return fom

def get_fom_vals(param_index, step_sizes, step_size_list, data_params, param_names, inv_covariance, mocks_dir = None, mocks_name = None):
    """
    """
    fom = np.zeros(len(step_sizes))
    for i, step_size in enumerate(step_sizes):
        step_size_list[param_index] = step_size
        temp_fisher = compute_stepsize_fisher(stepsizes = step_size_list, 
                                              deriv_params = param_names, 
                                              data_params = data_params, 
                                              inv_covariance = inv_covariance,
                                              mocks_dir = mocks_dir,
                                              mocks_name = mocks_name)
        fom[i] += calculate_fom(param_index = param_index, full_fisher_matrix = temp_fisher)
    
    return fom

def get_1d_fom_vals(step_sizes, data_params, param_names, inv_covariance, mocks_dir = None, mocks_name = None):
    """
    """
    fom = np.zeros(len(step_sizes))
    for i, step_size in enumerate(step_sizes):
        step_size_list = [step_size]
        temp_fisher = compute_stepsize_fisher(stepsizes = step_size_list, 
                                              deriv_params = param_names, 
                                              data_params = data_params, 
                                              inv_covariance = inv_covariance,
                                              mocks_dir = mocks_dir,
                                              mocks_name = mocks_name)
        fom[i] += 1/np.sqrt(temp_fisher)
    
    return fom

if __name__ == "__main__": 
    # step_sizes = np.array([])

    # for i in range(9):
    #     step_sizes = np.append(step_sizes, f"{0.00001 * (i+1):.5f}")
    # for i in range(9):
    #     step_sizes = np.append(step_sizes, f"{0.0001 * (i+1):.4f}")
    # for i in range(9):
    #     step_sizes = np.append(step_sizes, f"{0.001 * (i+1):.3f}")
    # for i in range(9):
    #     step_sizes = np.append(step_sizes, f"{0.01 * (i+1):.2f}")
    # step_sizes = np.append(step_sizes, 0.1)

    step_sizes = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1])

    # run_varying_stepsize(step_sizes = step_sizes, stencil_pts = 3, 
    #                      mocks_dir = "/share/data1/klin/kcap_out/kids_1000_cl_deriv_tests/kids_deriv_stepsize_test_3pt", mocks_name = "kids_stepsize",
    #                      deriv_ini_file = '/share/splinter/klin/kcap/runs/lfi_config/kids_cl_deriv_pipeline.ini', deriv_values_file = '/share/splinter/klin/kcap/runs/lfi_config/kids_cl_deriv_values.ini', 
    #                      deriv_values_list_file = '/share/splinter/klin/kcap/runs/lfi_config/kids_cl_deriv_values_list.ini', base_folder = '/share/data1/klin/kcap_out/kids_1000_cl_deriv_tests/template')
    run_varying_stepsize(step_sizes = step_sizes, stencil_pts = 5, 
                         mocks_dir = "/share/data1/klin/kcap_out/kids_1000_cl_deriv_tests/kids_deriv_stepsize_test_expanded", mocks_name = "kids_stepsize",
                         deriv_ini_file = '/share/splinter/klin/kcap/runs/lfi_config/kids_cl_deriv_pipeline.ini', deriv_values_file = '/share/splinter/klin/kcap/runs/lfi_config/kids_cl_deriv_values.ini', 
                         deriv_values_list_file = '/share/splinter/klin/kcap/runs/lfi_config/kids_cl_deriv_values_list.ini', base_folder = '/share/data1/klin/kcap_out/kids_1000_cl_deriv_tests/template')
    # run_varying_stepsize(step_sizes = step_sizes, stencil_pts = 7,
    #                      mocks_dir = "/share/data1/klin/kcap_out/kids_1000_cl_deriv_tests/kids_deriv_stepsize_test_7pt", mocks_name = "kids_stepsize",
    #                      deriv_ini_file = '/share/splinter/klin/kcap/runs/lfi_config/kids_cl_deriv_pipeline.ini', deriv_values_file = '/share/splinter/klin/kcap/runs/lfi_config/kids_cl_deriv_values.ini', 
    #                      deriv_values_list_file = '/share/splinter/klin/kcap/runs/lfi_config/kids_cl_deriv_values_list.ini', base_folder = '/share/data1/klin/kcap_out/kids_1000_cl_deriv_tests/template')
    # run_varying_stepsize(step_sizes = step_sizes, stencil_pts = 9, 
    #                      mocks_dir = "/share/data1/klin/kcap_out/kids_1000_cl_deriv_tests/kids_deriv_stepsize_test_9pt", mocks_name = "kids_stepsize",
    #                      deriv_ini_file = '/share/splinter/klin/kcap/runs/lfi_config/kids_cl_deriv_pipeline.ini', deriv_values_file = '/share/splinter/klin/kcap/runs/lfi_config/kids_cl_deriv_values.ini', 
    #                      deriv_values_list_file = '/share/splinter/klin/kcap/runs/lfi_config/kids_cl_deriv_values_list.ini', base_folder = '/share/data1/klin/kcap_out/kids_1000_cl_deriv_tests/template')
