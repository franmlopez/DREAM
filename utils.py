import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import erfi
from math import pi

def cdf(data, bins=10):
    data = data[data[:,1]==1]
    p=np.arange(100/bins, 100, 100/bins)
    cdf=np.percentile(data[:,0], p)
    return cdf

def delta(times_incongr, times_congr, bins=10):
    p = np.arange(100/bins, 100, 100/bins)
    diffs=np.percentile(times_incongr, p) - np.percentile(times_congr, p)
    means=(np.percentile(times_incongr, p)+np.percentile(times_congr, p))/2
    return means, diffs

def caf(data, bins=5):
    data = data[data[:,0].argsort()]
    trials_per_bin = int(np.floor(len(data[:,0]) / bins))
    caf = np.zeros(bins)
    for idx in range(bins):
        caf[idx] = np.mean(data[trials_per_bin*idx:trials_per_bin*(idx+1), 1])
    return caf

def plot_all_sim(caf_congr, caf_incongr, cdf_congr, cdf_incongr, save_name=None):
    fig, ax = plt.subplots(1,3, figsize=(16,4))

    #CAF
    percentiles = np.linspace(1/len(caf_congr), 1, len(caf_congr))
    ax[0].plot(percentiles, caf_congr, color='black', linewidth=1, label="Congruent")
    ax[0].plot(percentiles, caf_incongr, color='darkgray', linewidth=1, label="Incongruent")
    ax[0].set_ylabel('CAF')
    ax[0].set_xlabel('Time bin')
    ax[0].legend()
    #CDF
    percentiles = np.linspace(1/len(cdf_congr), 1, len(cdf_congr))
    ax[1].plot(cdf_congr, percentiles, color='black', linewidth=1, label="Congruent")
    ax[1].plot(cdf_incongr, percentiles, color='darkgray', linewidth=1, label="Incongruent")
    ax[1].set_ylabel('CDF')
    ax[1].set_xlabel('Time [ms]')
    #Delta
    exp_delta_x = (cdf_congr + cdf_incongr)/2
    exp_delta_y = cdf_incongr - cdf_congr
    ax[2].plot(exp_delta_x, exp_delta_y, linestyle='--', color='black', linewidth=1)
    ax[2].set_ylabel('Delta')
    ax[2].set_xlabel('Time [ms]')
    
    if save_name:
        fig.savefig('results/'+save_name+'.pdf', format='pdf', dpi=1200, bbox_inches='tight')

def plot_all_exp(caf_exp_congr, caf_exp_incongr, cdf_exp_congr, cdf_exp_incongr, save_name=None):
    fig, ax = plt.subplots(1,3, figsize=(16,4))

    #CAF
    percentiles = np.linspace(1/len(caf_exp_congr), 1, len(caf_exp_congr))
    ax[0].scatter(percentiles, caf_exp_congr, marker='o', s=50, facecolors='none', edgecolors='black', label='Congruent')
    ax[0].scatter(percentiles, caf_exp_incongr, marker='v', s=50, facecolors='none', edgecolors='black', label='Incongruent')
    ax[0].set_ylabel('CAF')
    ax[0].set_xlabel('Time bin')
    ax[0].legend()
    #CDF
    percentiles = np.linspace(1/len(cdf_exp_congr), 1, len(cdf_exp_congr))
    ax[1].scatter(cdf_exp_congr, percentiles, marker='o', s=50, facecolors='none', edgecolors='black', label='Congruent')
    ax[1].scatter(cdf_exp_incongr, percentiles, marker='v', s=50, facecolors='none', edgecolors='black', label='Incongruent')
    ax[1].set_ylabel('CDF')
    ax[1].set_xlabel('Time [ms]')
    #Delta
    exp_delta_x = (cdf_exp_congr + cdf_exp_incongr)/2
    exp_delta_y = cdf_exp_incongr - cdf_exp_congr
    ax[2].scatter(exp_delta_x, exp_delta_y, marker='D', s=50, facecolors='none', edgecolors='black')
    ax[2].set_ylabel('Delta')
    ax[2].set_xlabel('Time [ms]')

    if save_name:
        fig.savefig('results/'+save_name+'.pdf', format='pdf', dpi=1200, bbox_inches='tight')
    

def plot_all_fits(caf_exp_congr, caf_exp_incongr, caf_fit_congr, caf_fit_incongr,
              cdf_exp_congr, cdf_exp_incongr, cdf_fit_congr, cdf_fit_incongr,
              save_name=None):
    fig, ax = plt.subplots(1,3, figsize=(16,4))

    #CAF
    percentiles = np.linspace(1/len(caf_exp_congr), 1, len(caf_exp_congr))
    ax[0].scatter(percentiles, caf_exp_congr, marker='o', s=50, facecolors='none', edgecolors='black', label='Congruent, observed')
    ax[0].scatter(percentiles, caf_exp_incongr, marker='v', s=50, facecolors='none', edgecolors='black', label='Incongruent, observed')
    ax[0].plot(percentiles, caf_fit_congr, color='black', linewidth=1, label="Congruent, predicted")
    ax[0].plot(percentiles, caf_fit_incongr, color='darkgray', linewidth=1, label="Incongruent, predicted")
    ax[0].set_ylabel('CAF')
    ax[0].set_xlabel('Time bin')
    ax[0].legend()

    #CDF
    percentiles = np.linspace(1/len(cdf_exp_congr), 1, len(cdf_exp_congr))
    ax[1].scatter(cdf_exp_congr, percentiles, marker='o', s=50, facecolors='none', edgecolors='black', label='Congruent, observed')
    ax[1].scatter(cdf_exp_incongr, percentiles, marker='v', s=50, facecolors='none', edgecolors='black', label='Incongruent, observed')
    ax[1].plot(cdf_fit_congr, percentiles, color='black', linewidth=1, label="Congruent, predicted")
    ax[1].plot(cdf_fit_incongr, percentiles, color='darkgray', linewidth=1, label="Incongruent, predicted")
    ax[1].set_ylabel('CDF')
    ax[1].set_xlabel('Time [ms]')

    #Delta
    exp_delta_x = (cdf_exp_congr + cdf_exp_incongr)/2
    exp_delta_y = cdf_exp_incongr - cdf_exp_congr
    fit_delta_x = (cdf_fit_congr + cdf_fit_incongr)/2
    fit_delta_y = cdf_fit_incongr - cdf_fit_congr
    plt.scatter(exp_delta_x, exp_delta_y, marker='D', s=50, facecolors='none', edgecolors='black', label='Observed')
    ax[2].plot(fit_delta_x, fit_delta_y, linestyle='--', color='black', linewidth=1, label="Predicted")
    ax[2].set_ylabel('Delta')
    ax[2].set_xlabel('Time [ms]')
    ax[2].legend()
    
    if save_name:
        fig.savefig('results/'+save_name+'.pdf', format='pdf', dpi=1200, bbox_inches='tight')

def plot_activations(t, expected_X_c, expected_X_a_congr, expected_X_s_congr, expected_X_a_incongr, expected_X_s_incongr,
                     multi_X_s_congr, multi_X_s_incongr, save_name=None):
    fig, ax = plt.subplots(1,2, figsize=(16,4))

    # Expected activations
    ax[0].plot(t, expected_X_c, color='black', linestyle='dotted', label='Controlled')
    ax[0].plot(t, expected_X_a_congr, color='black', linestyle='--', label='Automatic, congruent')
    ax[0].plot(t, expected_X_a_incongr, color='darkgray', linestyle='--', label='Automatic, incongruent')
    ax[0].plot(t, expected_X_s_congr, color='black',  label='Superimposed, congruent')
    ax[0].plot(t, expected_X_s_incongr, color='darkgray',  label='Superimposed, incongruent')
    ax[0].set_xlabel('Time [ms]')
    ax[0].set_ylabel('Mean activations')
    ax[0].set_ylim(-np.max(expected_X_s_congr), np.max(expected_X_s_congr))
    ax[0].legend()

    # Examples
    for idx in range(multi_X_s_congr.shape[0]):
        ax[1].plot(t, multi_X_s_congr[idx,:], color='black', label='Congruent' if idx==0 else '_nolegend_')
        ax[1].plot(t, multi_X_s_incongr[idx,:], color='darkgray', label='Incongruent' if idx==0 else '_nolegend_')
    ax[1].set_xlabel('Time [ms]')
    ax[1].set_ylabel('Example activations')
    ax[1].set_ylim(-np.max(expected_X_s_congr), np.max(expected_X_s_congr))
    ax[1].legend()

    if save_name:
        fig.savefig('results/'+save_name+'.pdf', format='pdf', dpi=1200, bbox_inches='tight')

def plot_multi_delta(cdfs_congr, cdfs_incongr, labels, title, save_name=None):
    fig, ax = plt.subplots(figsize=(16/3,4))
    grays = ['#000000','#282828','#505050','#707070','#909090','#A9A9A9','#C0C0C0']
    
    for idx in range(7):
        fit_delta_x = (cdfs_congr[idx,:] + cdfs_incongr[idx,:])/2
        fit_delta_y = cdfs_incongr[idx,:] - cdfs_congr[idx,:]
        ax.plot(fit_delta_x, fit_delta_y, color=grays[idx], linewidth=1, label='%.1e' %labels[idx])
        ax.set_ylabel('Delta')
        ax.set_xlabel('Time [ms]')
        ax.set_title(title)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
    if save_name:
        fig.savefig('results/'+save_name+'.pdf', format='pdf', dpi=1200, bbox_inches='tight')