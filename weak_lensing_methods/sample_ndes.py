import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpcm
# import os
from pathlib import Path
# import errno
from getdist import plots, MCSamples
import plotting as pplot

def sample_single_NDEs(param_names, param_labels, n_ndes, DelfiEnsemble, results_directory, save_single_NDEs = False):
    nde_posterior_samples = []
    samples_for_plot = []
    Path(results_directory + '/saved_samples').mkdir(parents=True, exist_ok=True)
    param_priors = {}
    for i, name in enumerate(param_names):
        param_priors[name] = (DelfiEnsemble.prior.lower[i], DelfiEnsemble.prior.upper[i])
    for i in range (n_ndes+1):  # +1 to also include the stacked posterior
        if int(tf.__version__[0]) < 2:
            if i<n_ndes:
                log_like = lambda x:DelfiEnsemble.log_posterior_individual(i,x,DelfiEnsemble.data)
                posterior_samples, weights, _ = DelfiEnsemble.emcee_sample(log_target=log_like)
            else:              #stacked posterior
                posterior_samples, weights, _ = DelfiEnsemble.emcee_sample()
        else:
            if i<n_ndes:
                log_like = lambda x: DelfiEnsemble.weighted_log_posterior_individual(x, single_NDE=i)
                posterior_samples, weights, _ = DelfiEnsemble.emcee_sample(log_target=log_like)
            else:              #stacked posterior
                posterior_samples, weights, _ = DelfiEnsemble.emcee_sample()

        nde_posterior_samples.append(posterior_samples)

        if save_single_NDEs == True:
            if i<n_ndes:
                np.savetxt(results_directory + '/saved_samples/final_posterior_samples_NDE_{}.txt'.format(i), posterior_samples)
            else:
                np.savetxt(results_directory + '/saved_samples/final_posterior_samples.txt', posterior_samples)
        
        if i == n_ndes:
            nde_mc_samples = MCSamples(samples = posterior_samples,
                                       weights = weights,
                                       names = param_names, 
                                       labels = param_labels, 
                                       ranges = param_priors,
                                       label = 'NDEs')
        else:
            nde_mc_samples = MCSamples(samples = posterior_samples,
                                       weights = weights,
                                       names = param_names, 
                                       labels = param_labels, 
                                       ranges = param_priors,
                                       label = 'NDE_%s:'%i + ' with stacking weight of: ' + str(np.round_(DelfiEnsemble.stacking_weights[i], 3)))
        
        nde_mc_samples.saveAsText(root = results_directory + "/saved_samples/NDE_samples_"+str(i))
        samples_for_plot.append(nde_mc_samples)
        print("Finished sampling from NDE %s" % i)
    
    return nde_posterior_samples, samples_for_plot

def sample_likelihood(param_names, param_labels, DelfiEnsemble, theta, results_directory, label = 'Likelihood Samples', save_name = 'likelihood_samples', save_samples = False):
    likelihood_save_dir = results_directory + '/likelihood_samples/'
    Path(likelihood_save_dir).mkdir(parents=True, exist_ok=True)
    log_like = lambda x:DelfiEnsemble.log_likelihood_stacked(theta, x)
    likelihood_samples, weights, log_prob = DelfiEnsemble.emcee_sample(log_target=log_like) #, x0=theta

    if save_samples == True:
        np.savetxt(likelihood_save_dir+str(save_name)+'_likelihood_samples.txt', likelihood_samples)
        np.savetxt(likelihood_save_dir+str(save_name)+'_likelihood_weights.txt', weights)
        np.savetxt(likelihood_save_dir+str(save_name)+'_likelihood_log_prob.txt', log_prob)
        
        likelihood_mc_samples = MCSamples(samples = likelihood_samples,
                                       weights = weights,
                                       names = param_names, 
                                       labels = param_labels, 
                                       label = label)
        
        likelihood_mc_samples.saveAsText(root = likelihood_save_dir+str(save_name))

        print("Finished sampling")
    
    return likelihood_samples, weights, log_prob, likelihood_mc_samples

def evaluate_post_on_grid(n_ndes, DelfiEnsemble, x_vals, y_vals, results_dir, tol = 0.01, save_name = None):
    nde_mesh_samples = []
    tol = tol
    for nde in range(n_ndes+1):
        if nde == n_ndes:
            log_posterior = lambda x:DelfiEnsemble.log_posterior_stacked(x,DelfiEnsemble.data)
            log_posterior_mesh = np.zeros((100,100))
            for i, x in enumerate(x_vals):
                for j, y in enumerate(y_vals):
                        log_posterior_mesh[i][j] = log_posterior(np.array([x, y]))
            log_posterior_mesh = log_posterior_mesh.reshape((100,100)).T
            nde_mesh_samples.append(log_posterior_mesh)
            if save_name != None:
                np.savetxt(results_dir + save_name+'_stacked.txt', log_posterior_mesh)
        else:
            log_posterior = lambda x:DelfiEnsemble.log_posterior_individual(nde,x,DelfiEnsemble.data)    
            log_posterior_mesh = np.zeros((100,100))
            for i, x in enumerate(x_vals):
                for j, y in enumerate(y_vals):
                        log_posterior_mesh[i][j] = log_posterior(np.array([x, y]))
            log_posterior_mesh = log_posterior_mesh.reshape((100,100)).T
            nde_mesh_samples.append(log_posterior_mesh)
            if save_name != None:
                np.savetxt(results_dir + save_name+'_'+str(nde)+'.txt', log_posterior_mesh)

    return nde_mesh_samples