import numpy as np

import kcap_methods
import score_compression


def run_varying_stepsize():
    step_sizes = np.array([])
    for value in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
        for i in range(10):
            step_sizes = np.append(step_sizes, value * (i+1))

    for i, step_size in enumerate(step_sizes):
        kcap_methods.run_kcap_deriv(mock_run = i, 
                    param_to_vary = "cosmological_parameters--omch2", 
                    params_to_fix = ["cosmological_parameters--sigma_8", "intrinsic_alignment_parameters--a"],
                    vals_to_diff = ["shear_xi_minus_binned", "shear_xi_plus_binned", "theory_data_covariance"],
                    step_size = step_size)
        kcap_methods.run_kcap_deriv(mock_run = i, 
                    param_to_vary = "cosmological_parameters--sigma_8", 
                    params_to_fix = ["cosmological_parameters--omch2", "intrinsic_alignment_parameters--a"],
                    vals_to_diff = ["shear_xi_minus_binned", "shear_xi_plus_binned", "theory_data_covariance"],
                    step_size = step_size)
        kcap_methods.run_kcap_deriv(mock_run = i, 
                    param_to_vary = "intrinsic_alignment_parameters--a", 
                    params_to_fix = ["cosmological_parameters--omch2", "cosmological_parameters--sigma_8"],
                    vals_to_diff = ["shear_xi_minus_binned", "shear_xi_plus_binned", "theory_data_covariance"],
                    step_size = step_size)

def calculate_fom(mock_run, deriv_param, data_params):
    """
    Computes the fisher matrix based on the params set, 
    e.g params = ["shear_xi_plus_omch2_deriv", "shear_xi_plus_sigma_8_deriv", "shear_xi_minus_omch2_deriv", "shear_xi_minus_sigma_8_deriv"]
    """
    inv_covariance =  kcap_methods.get_inv_covariance(mock_run = mock_run)
    assert inv_covariance.shape[0] == inv_covariance.shape[1], "Square matrix for the inverse covariance not found"

    # deriv_vector = np.zeros(shape = (len(deriv_params), inv_covariance.shape[0]))
    deriv_vals_to_get = [data_param + '_' + deriv_param + '_deriv' for data_param in data_params]
    deriv_vector_dict = kcap_methods.get_values(mock_run = mock_run, vals_to_read = deriv_vals_to_get)
    deriv_vector = np.array([])
    for data_param in data_params:
        deriv_vector = np.append(deriv_vector, deriv_vector_dict[data_param+'_'+deriv_param+'_deriv'])

    fom = np.sqrt(np.dot(np.transpose(deriv_vector), np.dot(inv_covariance, deriv_vector)))
    return fom

def get_fom_vals(deriv_param, step_sizes):
    """
    """
    data_params = ['shear_xi_plus_binned', 'shear_xi_minus_binned']
    fom = np.zeros(len(step_sizes))
    for i, step_size in enumerate(step_sizes):
        fom[i] += calculate_fom(mock_run = str(i), deriv_param = deriv_param, data_params = data_params)
    
    return step_sizes, fom

if __name__ == "__main__":
    run_varying_stepsize()