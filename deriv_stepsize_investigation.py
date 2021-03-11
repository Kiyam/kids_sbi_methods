from . import deriv_run
import numpy as np

if __name__ == "__main__":
    covariance, inv_covariance = deriv_run.get_covariance(mock_run = 0)
    deriv_values = deriv_run.get_values(mock_run = 0, vals_to_read = ["shear_xi_plus", "shear_xi_minus", "shear_xi_plus_deriv"])