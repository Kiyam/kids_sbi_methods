import numpy as np
import scipy as sc
import os
import errno
import kcap_methods
import score_compression

# Some script to take a covariance and degrade it in some manner
# Ideas are to suppress the off diagonals, randomly change the magnitude of the on-diagonal components

def suppress_diagonals(covariance, suppression_type, suppression_factor):
    dim = len(covariance) # just gives number of rows and/or columns as expect a square matrix
    for row in range(dim):
        for column in range(dim):
            if suppression_type == "power_increasing":
                covariance[row][column] = covariance[row][column] * 10**(suppression_factor * abs(row - column))
            if suppression_type == "constant":
                covariance[row][column] = covariance[row][column] * suppression_factor * abs(np.sign(row - column))

    return covariance

def diagonal_wishart_noise(covariance, x, df):
    inverse_covariance = kcap_methods.calc_inverse_covariance(covariance)
    
    wishart_dist = sc.stats.wishart(x, df, inverse_covariance)

    return covariance

