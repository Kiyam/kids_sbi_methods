import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpcm
# import os
from pathlib import Path
# import errno
from getdist import plots, MCSamples

def sample_single_NDEs(param_names, n_ndes, DelfiEnsemble, results_directory, save_single_NDEs = False):
    nde_posterior_samples = []
    samples_for_plot = []
    Path(results_directory + '/saved_samples').mkdir(parents=True, exist_ok=True)
    for i in range (n_ndes+1):  # +1 to also include the stacked posterior
        if int(tf.__version__[0]) < 2:
            if i<n_ndes:
                log_like = lambda x:DelfiEnsemble.log_posterior_individual(i,x,DelfiEnsemble.data)
                posterior_samples, weights = DelfiEnsemble.emcee_sample(log_target=log_like)[0:2]
            else:              #stacked posterior
                posterior_samples, weights = DelfiEnsemble.emcee_sample()[0:2]
        else:
            if i<n_ndes:
                log_like = lambda x: DelfiEnsemble.weighted_log_posterior_individual(x, single_NDE=i)
                posterior_samples, weights = DelfiEnsemble.emcee_sample(log_target=log_like)[0:2]
            else:              #stacked posterior
                posterior_samples, weights = DelfiEnsemble.emcee_sample()[0:2]

        nde_posterior_samples.append(posterior_samples)

        if save_single_NDEs == True:
            if i<n_ndes:
                np.savetxt(results_directory + '/saved_samples/final_posterior_samples_NDE_{}.txt'.format(i), posterior_samples)
            else:
                np.savetxt(results_directory + '/saved_samples/final_posterior_samples.txt', posterior_samples)
        
        if i == n_ndes:
            nde_mc_samples = MCSamples(samples = posterior_samples,
                                       weights = weights,
                                       names = param_names, labels = param_names, label = 'NDEs')
        else:
            nde_mc_samples = MCSamples(samples = posterior_samples,
                                       weights = weights,
                                       names = param_names, labels = param_names, label = 'NDE_%s:'%i + ' with stacking weight of: ' + str(np.round_(DelfiEnsemble.stacking_weights[i], 3)))
        
        nde_mc_samples.saveAsText(root = results_directory + "/saved_samples/NDE_samples_"+str(i))
        samples_for_plot.append(nde_mc_samples)
        print("Finished sampling from NDE %s" % i)
    
    return nde_posterior_samples, samples_for_plot

def sample_likelihood(param_names, DelfiEnsemble, theta, results_directory, save_samples = False):
    Path(results_directory + '/saved_samples').mkdir(parents=True, exist_ok=True)
    log_like = lambda x:DelfiEnsemble.log_likelihood_stacked(theta, x)
    test_like = DelfiEnsemble.log_likelihood_stacked(theta, theta)
    likelihood_samples, weights, log_prob = DelfiEnsemble.emcee_sample(log_target=log_like) #, x0=theta

    if save_samples == True:
        np.savetxt(results_directory + '/saved_samples/likelihood_samples.txt', likelihood_samples)
        
        likelihood_mc_samples = MCSamples(samples = likelihood_samples,
                                       weights = weights,
                                       names = param_names, labels = param_names, label = 'NDEs')
        
        likelihood_mc_samples.saveAsText(root = results_directory + "/saved_samples/likelihood_samples")

        print("Finished sampling")
    
    return likelihood_samples, likelihood_mc_samples, log_prob