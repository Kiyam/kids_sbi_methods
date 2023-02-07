import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpcm
# import os
from pathlib import Path
# import errno
from getdist import plots, MCSamples

def triangle_plots(samples_for_plot, markers = None, filled = True, title = None, save_file = None, presentation = False):
    cm = mpcm.get_cmap('plasma')
    num_plots = len(samples_for_plot)
    colors = [cm(x) for x in np.linspace(0.01, 0.75, num_plots)] #sets up the plotting colours
    # Triangle plot
    
    g = plots.get_subplot_plotter(width_inch = 12)
    g.settings.solid_colors = colors
    g.settings.alpha_filled_add = 0.6
    
    if presentation is True:
        g.settings.axes_fontsize = 20
        g.settings.lab_fontsize = 30
        g.settings.legend_fontsize = 25
    else:
        g.settings.axes_fontsize = 20
        g.settings.lab_fontsize = 20
        g.settings.legend_fontsize = 20
    
    # g.settings.x_label_rotation=47
    # g.settings.legend_loc = 'center right'
    
    if filled == False:
        g.triangle_plot(samples_for_plot, filled=filled, normalized=True, markers = markers, legend_loc = 'upper right', contour_colors = colors)
    else:  
        g.triangle_plot(samples_for_plot, filled=filled, normalized=True, markers = markers, legend_loc = 'upper right')
    fig = g.fig
    if title is not None:
        fig.suptitle(title+"\n", va='bottom', fontsize = 30)
    if save_file is not None:
        fig.savefig(save_file, bbox_inches = 'tight')
        
def compressed_summary_plot(compressed_data, thetas, theta_names, theta_indices, save_file = None):
    fig, ax = plt.subplots(2, 2, figsize=(15,15))
    plt.xticks(fontsize = 16)
    fig.suptitle("Linearly score compressed data vs. parameters for a run varying %s and %s" % (theta_names[0], theta_names[1]), size=20, y = 0.99)

    im1 = ax[0, 0].scatter(thetas[:,theta_indices[0]], compressed_data[:,theta_indices[0]], c=thetas[:,theta_indices[1]])
    ax[0, 0].set_xlabel(theta_names[0], fontsize=20)
    ax[0, 0].set_ylabel('Compressed statistic %s' %(theta_indices[0]), labelpad = 15, fontsize=20)
    ax[0, 0].tick_params(axis='both', which='major', labelsize=15)
    cbar1 = fig.colorbar(im1, orientation='vertical', ax=ax[0, 0])
    cbar1.ax.tick_params(labelsize=15)
    cbar1.set_label(theta_names[1], fontsize=20)

    im2 = ax[1, 0].scatter(thetas[:,theta_indices[0]], compressed_data[:,theta_indices[1]], c=thetas[:,theta_indices[1]])
    ax[1, 0].set_xlabel(theta_names[0], fontsize=20)
    ax[1, 0].set_ylabel('Compressed statistic %s' %(theta_indices[1]), labelpad = 15, fontsize=20)
    ax[1, 0].tick_params(axis='both', which='major', labelsize=15)
    cbar2 = fig.colorbar(im2, orientation='vertical', ax=ax[1, 0])
    cbar2.ax.tick_params(labelsize=15)
    cbar2.set_label(theta_names[1], fontsize=20)

    im3 = ax[0, 1].scatter(thetas[:,theta_indices[1]], compressed_data[:,theta_indices[0]], c = thetas[:,theta_indices[0]])
    ax[0, 1].set_xlabel(theta_names[1], fontsize=20)
    ax[0, 1].set_ylabel('Compressed statistic %s' %(theta_indices[0]), labelpad = 15, fontsize=20)
    ax[0, 1].tick_params(axis='both', which='major', labelsize=15)
    cbar3 = fig.colorbar(im3, orientation='vertical', ax=ax[0, 1])
    cbar3.ax.tick_params(labelsize=15)
    cbar3.set_label(theta_names[0], fontsize=20)

    im4 = ax[1, 1].scatter(thetas[:,theta_indices[1]], compressed_data[:,theta_indices[1]], c = thetas[:,theta_indices[0]])
    ax[1, 1].set_xlabel(theta_names[1], fontsize=20)
    ax[1, 1].set_ylabel('Compressed statistic %s' %(theta_indices[1]), labelpad = 15, fontsize=20)
    ax[1, 1].tick_params(axis='both', which='major', labelsize=15)
    cbar4 = fig.colorbar(im4, orientation='vertical', ax=ax[1, 1])
    cbar4.ax.tick_params(labelsize=15)
    cbar4.set_label(theta_names[0], fontsize=20)
        
    plt.tight_layout(pad = 2.0)
    if save_file is not None:
        fig.savefig(save_file)
    
    plt.show()
    
def compressed_summary_plot_large(compressed_data, thetas, theta_names, save_file = None):
    num_params = len(theta_names)
    fig, ax = plt.subplots(num_params, num_params, figsize = (40, 40))
    plt.xticks(fontsize = 16)
    fig.suptitle("Linearly score compressed data vs. parameters", size=20, y = 0.99)
    
    for i in range(num_params):
        for j in range(num_params):
            # ax[i, j].scatter(thetas[:,j], compressed_data[:,i], c=thetas[:,j])
            ax[i, j].scatter(thetas[:,j], compressed_data[:,i])
            ax[i, j].set_xlabel(theta_names[j], fontsize=15)
            ax[i, j].set_ylabel('Compressed statistic %s' %(i), labelpad = 5, fontsize=15)
            ax[i, j].tick_params(axis='both', which='major', labelsize=15)
        
    plt.tight_layout(pad = 2.0)
    if save_file is not None:
        fig.savefig(save_file)
    
    plt.show()
    
def compressed_summary_plot_small(compressed_data, thetas, theta_names, theta_indices, save_file = None):
    fig, ax = plt.subplots(1, 2, figsize=(16,8))
    plt.xticks(fontsize = 16)
    fig.suptitle("Linearly score compressed data vs. parameters", size=20, y = 0.99)

    im1 = ax[0].scatter(thetas[:,theta_indices[0]], compressed_data[:,theta_indices[0]], c=thetas[:,theta_indices[1]])
    ax[0].set_xlabel('Parameter %s' %(theta_indices[0]+1), fontsize=20)
    ax[0].set_ylabel('Compressed statistic %s' %(theta_indices[0]+1), labelpad = 15, fontsize=20)
    ax[0].tick_params(axis='both', which='major', labelsize=15)
    cbar1 = fig.colorbar(im1, orientation='vertical', ax=ax[0])
    cbar1.ax.tick_params(labelsize=15)
    cbar1.set_label('Parameter %s' %(theta_indices[1]+1), fontsize=20)

    im2 = ax[1].scatter(thetas[:,theta_indices[1]], compressed_data[:,theta_indices[1]], c = thetas[:,theta_indices[0]])
    ax[1].set_xlabel('Parameter %s' %(theta_indices[1]+1), fontsize=20)
    ax[1].set_ylabel('Compressed statistic %s' %(theta_indices[1]+1), labelpad = 15, fontsize=20)
    ax[1].tick_params(axis='both', which='major', labelsize=15)
    cbar2 = fig.colorbar(im2, orientation='vertical', ax=ax[1])
    cbar2.ax.tick_params(labelsize=15)
    cbar2.set_label('Parameter %s' %(theta_indices[0]+1), fontsize=20)
        
    plt.tight_layout(pad = 2.0)
    if save_file is not None:
        fig.savefig(save_file)
    
    plt.show()