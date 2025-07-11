#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains all visualization functions.
"""

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import uniform

custom_palette = ["blue", "orange", "green", "gray"]

sns.set_palette(custom_palette)


    
def plot_loss_against_rho(kls, pis, maxK, lam, fig_name, log=True,legend=True,rho_min = 0, rho_max = 10):
    r = np.linspace(rho_min,rho_max,100)
    fig, axis = plt.subplots(1, 1, figsize=(6,3))
    for pi, kl, K in zip(pis, kls, np.arange(1,maxK)):
        y = np.sum(np.multiply(pi, np.maximum(kl-r[:,np.newaxis],0)),axis = 1) + lam * K
        if K==2: ypos = min(y)
        axis.plot(r, y, lw=1.8,label = "K=%i"%K)
    xpos = np.max(kls[1])+(kls[0][0]-np.max(kls[1]))/2
    print(xpos)
    #axis.axvline(x=1.1616, c='black', ls='dashed',lw=1)
    # axis.legend(loc = 'upper center',fontsize = 20, bbox_to_anchor=(1.15, .9), fancybox=True, shadow=True)

    axis.set_xlabel(r'$\rho$',fontsize=20)
    axis.set_ylabel('Penalized loss',fontsize=20)
    axis.annotate('X', xy=(xpos, ypos), color='black', fontsize=14, ha='center', va='center', weight='bold')
    axis.annotate(r'$K = 2$', xy=(xpos, 0.6*ypos), fontsize=13, ha='center', va='center')

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
    # axis.arrow(0.35, 0.0022, 0.2, 0.001, fc='black', ec='black')

    if legend:
        axis.legend(prop={'size': 12})
        fig.savefig(fig_name+'-legend.pdf',bbox_inches='tight')    
    else:
        axis.legend('', frameon=False)
        fig.savefig(fig_name+'.pdf',bbox_inches='tight')  
    
    
def plot_cluster_by_iteration(K, z, x, fig_name):
    np.random.seed(233)
    y = uniform.rvs(size = len(x))
    fig, axis = plt.subplots(1, 1, figsize=(8,5))
    axis.scatter(x, y, c= z, marker = ".")
    axis.set_yticks([])
    fig.tight_layout(h_pad=1)
    fig.savefig(fig_name+'-cluster-plot-K=%i.png'%(K))
    

   