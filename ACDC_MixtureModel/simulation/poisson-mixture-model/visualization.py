#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import uniform,poisson
from analysis import F_measure

custom_palette = ["blue", "orange", "green", "gray"]

sns.set_palette(custom_palette)


def plot_true_cluster(x, z_t, figname):
    np.random.seed(233)
    y = uniform.rvs(size = len(x))
    K_t = len(np.unique(z_t))
    fig, axis = plt.subplots(1, 1, figsize=(10,3))
    axis.scatter(x, y, c= z_t, marker = ".")
    axis.set_title( 'Truth, $K = %.f$'%K_t, fontsize= 16)
    axis.set_yticks([])
    axis.xaxis.set_tick_params(labelsize=16)
    fig.tight_layout(h_pad=1)
    fig.savefig(figname)
    
    
def plot_true_density(data, figname, xmin = 0, xmax = 200):
    grid = np.arange(xmin, xmax, 0.01)
    truth = np.zeros(len(grid))  
    for k in range(data.ncomp()):
        truth += data.pmf(grid, k)
    
    fig, axis = plt.subplots(1, 1, figsize=(8,5))
    axis.plot(grid, truth, 'red')  
    axis.legend(["True distribution"],prop={'size': 16})
    axis.set_xlabel("X", size = 20)
    axis.set_ylabel("Density", size = 20)
    axis.xaxis.set_tick_params(labelsize=16)
    axis.yaxis.set_tick_params(labelsize=16)
    fig.tight_layout(h_pad=1)
    sns.despine()
    fig.savefig(figname)
    
    

def plot_fmeasure_against_K(z_sims, z_true, K_true, fig_name):
    len_K = len(z_sims)
    f = np.zeros(len_K)
    for k in range(len_K):
        f[k] = F_measure(z_true,z_sims[k])

    fig, axis = plt.subplots(1, 1, figsize=(8,5))

    axis.plot(np.arange(1,len_K+1), f, label = 'ACDC')
    axis.set_xlabel(r'$K$',fontsize=20)
    axis.set_ylabel('F-measure',fontsize=20)
    axis.vlines(x = K_true, label = 'Truth', ymin=min(f), ymax=max(f), color='r', ls = '--')
    axis.set_ylim([0.9*min(f), 1])
    axis.xaxis.set_tick_params(labelsize=16)
    axis.yaxis.set_tick_params(labelsize=16)
    sns.despine()
#     axis.hlines(y = 1, xmin=1, xmax=len_K, color='gray', ls = '--')
    axis.legend(loc = 'lower right',prop={'size': 16})
    fig.savefig(fig_name)    
    return f



def plot_cluster_by_iteration(K, z, x, fig_name):
    np.random.seed(233)
    y = uniform.rvs(size = len(x))
    fig, axis = plt.subplots(1, 1, figsize=(10,3))
    axis.scatter(x, y, c = z, marker = ".")
    axis.set_title("ACDC, K = %i"%(K),fontsize= 14)
    axis.set_yticks([])
   
    fig.tight_layout(h_pad=1)
    fig.savefig(fig_name+'-cluster-plot-K=%i.png'%(K))
    


def plot_loss_against_rho(kls, pis, K_true, maxK, lam, fig_name, log=True,legend=True,rho_min = 0, rho_max = 10):
    r = np.linspace(rho_min,rho_max,100)
    fig, axis = plt.subplots(1, 1, figsize=(6,3))
    for pi, kl, K in zip(pis, kls, np.arange(1,maxK)):
    # for pi, kl, K in zip(pis, kls, np.arange(K_true-2,K_true+3)):
        y = np.sum(np.multiply(pi, np.maximum(kl-r[:,np.newaxis],0)),axis = 1) + lam * K
        if K==K_true: 
            ypos = min(y) 
            print(y)
        axis.plot(r, y, lw=1.8,label = "K=%i"%K)
    xpos = np.max(kls[K_true-1])+(np.max(kls[K_true-2])-np.max(kls[K_true-1]))/2

    axis.set_xlabel(r'$\rho$',fontsize=20)
    axis.set_ylabel('Penalized loss',fontsize=20)
    axis.annotate('X', xy=(xpos, ypos), color='black', fontsize=13, ha='center', va='center', weight='bold')
    axis.annotate(r'$K = %i$'%K_true, xy=(xpos, 0.6*ypos), fontsize=13, ha='center', va='center')

    if log:
        axis.set_yscale('log')
    axis.xaxis.set_tick_params(labelsize=13)
    axis.yaxis.set_tick_params(labelsize=13)
    sns.despine()
    if legend:
        axis.legend(prop={'size': 12})
        fig.savefig(fig_name+'-legend.pdf',bbox_inches='tight')    
    else:
        axis.legend('', frameon=False)
    sns.despine()

    if legend:
        axis.legend(prop={'size': 12})
        fig.savefig(fig_name+'-legend.pdf',bbox_inches='tight')    
    else:
        axis.legend('', frameon=False)
        fig.savefig(fig_name+'.pdf',bbox_inches='tight')  
    