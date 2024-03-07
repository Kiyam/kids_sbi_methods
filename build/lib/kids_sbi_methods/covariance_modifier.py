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
    