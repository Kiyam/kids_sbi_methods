import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import norm

def calc_params(sub_fisher): 
    part_1 = 0.5*(sub_fisher[0][0] + sub_fisher[1][1])
    part_2 = np.sqrt(0.25*(sub_fisher[0][0] - sub_fisher[1][1])**2 + sub_fisher[0][1]**2)
    a2 = part_1 + part_2 
    b2 = part_1 - part_2
    a = np.sqrt(a2)
    b = np.sqrt(b2)
    theta = - np.arctan2(2* sub_fisher[0][1], sub_fisher[0][0] - sub_fisher[1][1])/2 * (180 / np.pi)
    # t2theta = 2 * sub_fisher[0][1]/(sub_fisher[0][0] - sub_fisher[1][1])
    # theta = np.arctan2(t2theta)/2 * (180 / np.pi) 
    return a, b, theta

def plot_fisher_contours(height, width, angle, fiducial_x, fiducial_y, x_label, y_label, x_lims, y_lims):
    confidence_coeffs = [1.52, 2.48, 3.44]
    colors = ['#CCF1FF', '#E0D7FF', '#FFCCE1']
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    ells = [Ellipse(xy=(fiducial_x, fiducial_y), 
                    width=confidence_coeffs[i]*width, 
                    height=confidence_coeffs[i]*height, 
                    angle = angle, 
                    edgecolor=colors[i], fc='None', lw=3, ls = '-', alpha = None) for i in range(3)]
    
    plt.xlim(x_lims)
    plt.ylim(y_lims)
    for e in ells:
        ax.add_artist(e)
    plt.plot(fiducial_x, fiducial_y, 'x')
    plt.grid(color='lightgray',linestyle='--')
    ax.set_title(x_label + ' vs. ' + y_label + ' Fisher Contours')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label, labelpad = 15)
    plt.show()

def calc_and_plot(param_1_index, param_2_index, param_names, fisher_matrix, fiducial_params, lower_prior = None, upper_prior = None):
    sub_cov = np.linalg.inv(fisher_matrix[np.ix_([param_1_index,param_2_index],[param_1_index,param_2_index])])
    a, b, theta = calc_params(sub_cov)
    print(a, b, theta)
    if lower_prior is None:
        y_lim_lower = min(fiducial_params[param_2_index] - 2*a/(np.cos(theta)), fiducial_params[param_2_index] - 2*b/(np.cos(theta)))
        y_lim_max = max(fiducial_params[param_2_index] + 2*a/(np.cos(theta)), fiducial_params[param_2_index] + 2*b/(np.cos(theta)))
        x_lim_lower = min(fiducial_params[param_1_index] - 2*b/(np.sin(theta)),fiducial_params[param_1_index] - 2*a/(np.sin(theta)))
        x_lim_max = max(fiducial_params[param_1_index] + 2*b/(np.sin(theta)), fiducial_params[param_1_index] + 2*a/(np.sin(theta)))
    else:
        y_lim_lower = lower_prior[param_2_index]
        y_lim_max = upper_prior[param_2_index]
        x_lim_lower = lower_prior[param_1_index]
        x_lim_max = upper_prior[param_1_index]
    plot_fisher_contours(height = a, width = b, angle = theta,
                        fiducial_x = fiducial_params[param_1_index], fiducial_y = fiducial_params[param_2_index],
                        x_label = param_names[param_1_index], y_label = param_names[param_2_index],
                        x_lims = [x_lim_lower, x_lim_max], y_lims = [y_lim_lower, y_lim_max])



def triangle_plot(param_names, fisher_matrix, fiducial_params, lower_prior, upper_prior, results_directory = None):
    confidence_coeffs = [1.52, 2.48, 3.44]
    colors = ['#CCF1FF', '#E0D7FF', '#FFCCE1']
    
    num_params = len(param_names)
    fig, ax = plt.subplots(num_params, num_params, gridspec_kw = {'wspace':0, 'hspace':0}, figsize = (15,15))
    # fig.suptitle('Fisher Contours for Trial 14', fontsize=16)
    for j in range(num_params):
        for i in range(num_params):
            if i != j:
                if i > j:
                    ax[j, i].set_visible(False)
                else:
                    sub_cov = np.linalg.inv(fisher_matrix[np.ix_([j,i],[j,i])])
                    a, b, theta = calc_params(sub_cov)
                    y_mean = fiducial_params[j]
                    x_mean = fiducial_params[i]
                    
                    ells = [Ellipse(xy=(x_mean, y_mean), 
                        width=confidence_coeffs[k]*b, 
                        height=confidence_coeffs[k]*a, 
                        angle = theta, 
                        edgecolor=colors[k], fc='None', lw=3, ls = '-', alpha = None) for k in range(3)]
                    
                    for e in ells:
                        ax[j, i].add_artist(e)
                    
                    ax[j, i].plot(x_mean, y_mean, 'x')
                    ax[j, i].grid(color='lightgray',linestyle='--')
                    
                    plot_scf = 4
                    xmin, xmax = ax[j, i].get_xlim()
                    ymin, ymax = ax[j, i].get_ylim()
                    ax[j, i].set_ylim(y_mean - abs(y_mean - ymin) * plot_scf, 
                                      y_mean + abs(y_mean - ymax) * plot_scf)
                    ax[j, i].set_xlim(x_mean - abs(x_mean - xmin) * plot_scf, 
                                      x_mean + abs(x_mean - xmax) * plot_scf)
            else:
                std = np.sqrt(1/fisher_matrix[i][j])
                mean = fiducial_params[i]
                ax[j, i].set_xlim(lower_prior[i], upper_prior[i])
                x_vals = np.linspace(lower_prior[i], upper_prior[i], 100)
                y_vals = norm(mean, std)
                ax[j, i].plot(x_vals, y_vals.pdf(x_vals))
            
            if i == 0 and j == 0:
                ax[j, i].tick_params(which='both',
                                     bottom = False,
                                     top = False,
                                     right = False,
                                     left = False,
                                     labelbottom = False,
                                     labeltop = False,
                                     labelright = False,
                                     labelleft = False)
            elif j == (num_params - 1) and i == 0:
                ax[j, i].set_xticks(ax[j, i].get_xticks()[1:-1])
                ax[j, i].set_yticks(ax[j, i].get_yticks()[1:-1])
                ax[j, i].tick_params(which='both',
                                     bottom = False,
                                     top = False,
                                     right = False,
                                     labeltop = False,
                                     labelright = False)
                ax[j, i].set_xlabel(param_names[i], fontsize = 20)
                ax[j, i].set_ylabel(param_names[j], fontsize = 20)
                
            elif i == 0:
                ax[j, i].set_yticks(ax[j, i].get_yticks()[1:-1])
                ax[j, i].tick_params(which='both',
                                     bottom = False,
                                     top = False,
                                     right = False,
                                     labelbottom = False,
                                     labeltop = False,
                                     labelright = False)
                ax[j, i].set_ylabel(param_names[j], fontsize = 20)
                
            elif j == (num_params - 1):
                ax[j, i].set_xticks(ax[j, i].get_xticks()[1:-1])
                ax[j, i].tick_params(which='both',
                                     left = False,
                                     top = False,
                                     right = False,
                                     labeltop = False,
                                     labelright = False,
                                     labelleft = False)
                ax[j, i].set_xlabel(param_names[i], fontsize = 20)
            else:
                ax[j, i].tick_params(axis = 'both',
                                    which = 'both',
                                    left = False,
                                    top = False,
                                    bottom = False,
                                    right = False,
                                    labelbottom = False,
                                    labeltop = False,
                                    labelright = False,
                                    labelleft = False)
    
    if results_directory is not None:
        plt.savefig("{}/fisher_contours.png".format(results_directory), bbox_inches = 'tight', pad_inches = 0.2)