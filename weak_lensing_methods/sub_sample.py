import numpy as np

def find_most_even(lower, upper, params, sub_sample_size = 100, iteration_num = 10000):
    centers = np.zeros((sub_sample_size, len(lower)))
    for i in range(len(lower)):
        cut = np.linspace(lower[i], upper[i], sub_sample_size + 1)
        a = cut[:sub_sample_size]
        b = cut[1:sub_sample_size + 1]
        center = (a + b)/2
        centers[:,i] = center

    idx = np.random.randint(len(params), size=sub_sample_size) #size defines the number of values to fetch
    best_idx = idx

    new_sub_sample = params[idx,:]
    best_sub_sample = new_sub_sample
    distance = 0
    for i in range(len(lower)):
        distance += sum(np.abs((np.abs(sorted(new_sub_sample[:,i]) - centers[:,i])/centers[:,i])))
    iterations = 0

    while iterations <= iteration_num:
        idx = np.random.randint(len(params), size=sub_sample_size) #size defines the number of values to fetch
        new_sub_sample = params[idx,:]
        total_dist = 0

        for i in range(len(lower)):
            total_dist += sum(np.abs((np.abs(sorted(new_sub_sample[:,i]) - centers[:,i])/centers[:,i])))

        if total_dist < distance:
            distance = total_dist
            best_idx = idx
            best_sub_sample = new_sub_sample
        
        iterations += 1
    
    return best_sub_sample, best_idx

if __name__ == "__main__":
    lower = np.array([0.6, 0.07, -6., 0.8, 2.0, 0.6, 0.019, -5.0, -5.0, -5.0, -5.0, -5.0])
    upper = np.array([1.0, 0.18, 6., 1.15, 4.0, 0.9, 0.026, 5.0, 5.0, 5.0, 5.0, 5.0])
    params = np.loadtxt('/home/klin/pydelfi_out/weak_lensing/trial_44_sim_num_test/compressed_data_16000_fisher/thetas.dat')

    delta_z_cov = np.genfromtxt('/home/klin/Cat_to_Obs_K1000_P1/data/kids/nofz/SOM_cov_multiplied.asc')
    inv_L = np.linalg.inv(np.linalg.cholesky(delta_z_cov))
    unmixed_thetas = params
    corr_nz = params[7:, 7:]
    uncorr_nz = np.inner(inv_L, corr_nz).T
    unmixed_thetas[7:, 7:] = uncorr_nz

    test, idx = find_most_even(lower, upper, unmixed_thetas, sub_sample_size = 12000, iteration_num=10000)