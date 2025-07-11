#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from analysis import F_measure

def plot_loss_against_rho(kls, pis, maxK, lam, fig_name, log=True,rho_min = 0, rho_max = 10):
    r = np.linspace(rho_min,rho_max,100)
    fig, axis = plt.subplots(1, 1, figsize=(6,3))
    for pi, kl, K in zip(pis, kls, np.arange(1,maxK)):
        y = np.sum(np.multiply(pi, np.maximum(kl-r[:,np.newaxis],0)),axis = 1) + lam * K
        axis.plot(r, y, label = "K=%i"%K)
    #axis.axvline(x=1.1616, c='black', ls='dashed',lw=1)
    # axis.legend(loc = 'upper center',fontsize = 20, bbox_to_anchor=(1.15, .9), fancybox=True, shadow=True)
    axis.legend(prop={'size': 14})   
    axis.set_xlabel(r'$\rho$',fontsize=20)
    axis.set_ylabel('Penalized loss',fontsize=20)
    if log:
        axis.set_yscale('log')
    axis.xaxis.set_tick_params(labelsize=16)
    axis.yaxis.set_tick_params(labelsize=16)
    sns.despine()
    fig.savefig(fig_name+'.pdf',bbox_inches='tight')  
    


def plot_fmeasure_against_K(z_sims, z_true, K_true, fig_name):
    len_K = len(z_sims)
    f = np.zeros(len_K)
    for k in range(len_K):
        f[k] = F_measure(z_true,z_sims[k])

    fig, axis = plt.subplots(1, 1, figsize=(8,6))

    axis.plot(np.arange(1,len_K+1), f, label = 'ACDC')
    axis.set_xlabel(r'$K$',fontsize=20)
    axis.set_ylabel('F-measure',fontsize=20)
    axis.vlines(x = K_true, label = 'Truth', ymin=min(f), ymax=max(f), color='r', ls = '--')
    axis.legend(prop={'size': 20})   
    axis.xaxis.set_tick_params(labelsize=16)
    axis.yaxis.set_tick_params(labelsize=16)
    sns.despine()
    fig.savefig(fig_name+'.pdf',bbox_inches='tight')  
   
    
  


  