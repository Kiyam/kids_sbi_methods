import numpy as np

import kcap_methods
import score_compression


def run_varying_stepsize():
    for i, step_size in enumerate(np.linspace(0.0001, 0.1, 20)):
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

if __name__ == "__main__":
    run_varying_stepsize()