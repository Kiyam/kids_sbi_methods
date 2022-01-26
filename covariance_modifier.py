import numpy as np
import kcap_methods
from pathlib import Path
from scipy.stats import wishart

# Some script to take a covariance and degrade it in some manner
# Ideas are to suppress the off diagonals, randomly change the magnitude of the on-diagonal components

def suppress_diagonals(covariance, suppression_type, suppression_factor):
    dim = len(covariance) # just gives number of rows and/or columns as expect a square matrix
    for row in range(dim):
        for column in range(dim):
            if suppression_type == "power_increasing":
                covariance[row][column] = covariance[row][column] * 10**(-1 * suppression_factor * abs(row - column))
            if suppression_type == "constant":
                covariance[row][column] = covariance[row][column] * suppression_factor * abs(np.sign(row - column))

    return covariance

def wishart_noise(covariance, df):

    sampled_covariance = wishart.rvs(df, covariance)/df

    return sampled_covariance

if __name__ == "__main__":
    covariance = kcap_methods.get_covariance(mock_run = 0, which_cov = "covariance", mocks_dir = "/share/data1/klin/kcap_out/kids_fiducial_data_mocks", mocks_name = "kids_1000_cosmology_fiducial")
    file_location = "/share/data1/klin/kcap_out/kids_fiducial_data_mocks/varied_covariances_wishart_batch"

    # for factor in np.arange(1.0, 2.1, 0.1):
    #     new_covariance = suppress_diagonals(covariance = covariance, suppression_type = "power_increasing", suppression_factor = factor)
    #     outfile = file_location + '/power_increasing_covariance_' + str(factor) + '.dat'
    #     Path(file_location).mkdir(parents=True, exist_ok=True)
    #     np.savetxt(outfile, new_covariance)
    
    for factor in np.geomspace(300, 10000, 15):
        repeat_factor = 5
        for i in range(repeat_factor):
            df = int(factor)
            new_covariance = wishart_noise(covariance = covariance, df = df)
            outfile = file_location + '/wishart_' + str(df) + '_' + str(i) + '.dat'
            Path(file_location).mkdir(parents=True, exist_ok=True)
            np.savetxt(outfile, new_covariance)