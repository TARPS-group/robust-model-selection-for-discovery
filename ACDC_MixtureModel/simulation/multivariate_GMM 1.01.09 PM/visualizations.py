#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 10:37:19 2023

@author: jwli
"""

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


custom_palette = ["blue", "orange", "green", "gray"]

sns.set_palette(custom_palette)



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
    
    
def plot_true_cluster(D, K_true, x, z, fig_name):
    fig, axis = plt.subplots(3,3, figsize=(20,6), sharex = 'col', sharey = True)
    axis_0 = 0
    axis_1 = 0
    for i in range(D-1):
        axis[axis_0,axis_1].scatter(x[:,i], x[:,i+1], c = z, alpha = 0.7, marker = ".", s = 10)
        axis[axis_0,axis_1].set_title("Dimension%.f.H vs Dimension%.f.H (Manual, K = %i)"%(i+1, (i+2), K_true), fontsize=18)
        if (i+1)%3 == 0:
            axis_0 +=1
            axis_1 = 0
        else: 
            axis_1 += 1
    fig.tight_layout(h_pad=1)
    fig.savefig(fig_name+'-cluster-plot-K=%i.png'%(K_true))
    
def plot_cluster(D, z, x, title, fig_name):
    fig, axis = plt.subplots(3,3, figsize=(20,6), sharex = 'col', sharey = True)
    axis_0 = 0
    axis_1 = 0
    for i in range(D-1):
        axis[axis_0,axis_1].scatter(x[:,i], x[:,i+1], c = z, alpha = 0.7, marker = ".", s = 10)
        axis[axis_0,axis_1].set_title("Dimension%.f.H vs Dimension%.f.H (ACDC)"%(i+1, (i+2)), fontsize=18)
        if (i+1)%3 == 0:
            axis_0 +=1
            axis_1 = 0
        else: 
            axis_1 += 1
    fig.tight_layout(h_pad=1)
    fig.savefig(fig_name+'-%s.png'%title)
    
    
    
    
    
    
    