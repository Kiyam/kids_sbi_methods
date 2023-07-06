import numpy as np

def get_hist(samples, n_bins):
    counts, bin_edges = np.histogram(samples, bins=n_bins)
    probs = counts/len(samples)
    return probs, bin_edges

def find_marginal_mass(samples, probs, bin_edges):
    max_sample = max(samples)
    min_sample = min(samples)
    mask = np.array(np.where((bin_edges>min_sample) & (bin_edges<=max_sample))).flatten()
    mask = np.concatenate((np.array([mask.min() - 1]), mask))
    if max_sample >= bin_edges[-1]:
        kept_probs = probs[np.where((bin_edges>min_sample))[:-1]] #This exists purely for the counting as there are 1 more bin edges than actual bins
    else:
        kept_probs = probs[np.where((bin_edges>min_sample) & (bin_edges<=max_sample))]
    return kept_probs.sum()

def find_sigma(samples, probs, bin_edges, alpha):
    for i in range(len(samples)):
        if i < 100:
            pass
        else:
            temp_samples = samples[:i+1]
            probability = find_marginal_mass(temp_samples, probs, bin_edges)
            if probability >= alpha:
                return temp_samples.min(), temp_samples.max()
    
def resample_params(nde_samples):
    omega_m_nde_samples = np.copy(nde_samples)
    omega_m_nde_samples[:,1] = (nde_samples[:,1] + nde_samples[:,6])/(nde_samples[:,5]**2)
    omega_m_nde_samples[:,6] = (nde_samples[:,6])/(nde_samples[:,5]**2)
    return omega_m_nde_samples

def calc_pjhpd(samples, probs, n_bins = 1000):
    """
    Returns the sigma values in the order of -2sigma, -1sigma, +1sigma, +2sigma
    """
    num_params = len(samples[0])
    probs_sorted = probs.argsort()
    sorted_samples = samples[probs_sorted[::-1]]
    
    sigmas = np.zeros((num_params,4))
    for i in range(num_params):
        temp_samples = sorted_samples[:,i]
        probs, bin_edges = get_hist(temp_samples, n_bins=1000)
        minval_1, maxval_1 = find_sigma(temp_samples, probs, bin_edges, alpha = 0.68)
        minval_2, maxval_2 = find_sigma(temp_samples, probs, bin_edges, alpha = 0.95)
        sigmas[i] = np.array([minval_2, minval_1, maxval_1, maxval_2])
        # print("{0} 1 sigma range: {1}, {2}, 2 sigma range: {3}, {4}".format(resampled_param_names[i], minval_1, maxval_1, minval_2, maxval_2))  
    
    return sigmas