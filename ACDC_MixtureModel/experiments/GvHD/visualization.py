#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from analysis import  F_measure

# custom_palette = ["blue", "orange", "green", "gray", 'pink']

# sns.set_palette(custom_palette)
sns.set_palette('colorblind')

def plot_true_cluster(s, z_t, figname):
    K_true = len(np.unique(z_t))
    fig, axis = plt.subplots(1, 3, figsize=(20,6))
    axis[0].scatter(s.loc[:,"FL2.H"], s.loc[:,"FL1.H"], c = z_t, alpha = 0.7, marker = ".", s = 10)
    axis[0].set_title("FL1.H vs FL2.H (Manual, K = %.f)"%K_true, fontsize=20)
    axis[1].scatter(s.loc[:,"FL3.H"], s.loc[:,"FL2.H"], c = z_t, alpha = 0.7, marker = ".", s = 10)
    axis[1].set_title("FL2.H vs FL3.H (Manual, K = %.f)"%K_true, fontsize=20)
    axis[2].scatter(s.loc[:,"FL4.H"], s.loc[:,"FL3.H"], c = z_t, alpha = 0.7, marker = ".", s = 10)
    axis[2].set_title("FL3.H vs FL4.H (Manual, K = %.f)"%K_true, fontsize=20)
    fig.tight_layout(h_pad=1)
    fig.savefig(figname)

def plot_em_cluster(s, z_t, figname):
    K_true = len(np.unique(z_t))
    fig, axis = plt.subplots(1, 3, figsize=(20,6))
    axis[0].scatter(s.loc[:,"FL2.H"], s.loc[:,"FL1.H"], c = z_t, alpha = 0.7, marker = ".", s = 10)
    axis[0].set_title("FL1.H vs FL2.H (EM, K = %.f)"%K_true, fontsize=20)
    axis[1].scatter(s.loc[:,"FL3.H"], s.loc[:,"FL2.H"], c = z_t, alpha = 0.7, marker = ".", s = 10)
    axis[1].set_title("FL2.H vs FL3.H (EM, K = %.f)"%K_true, fontsize=20)
    axis[2].scatter(s.loc[:,"FL4.H"], s.loc[:,"FL3.H"], c = z_t, alpha = 0.7, marker = ".", s = 10)
    axis[2].set_title("FL3.H vs FL4.H (EM, K = %.f)"%K_true, fontsize=20)
    fig.tight_layout(h_pad=1)
    fig.savefig(figname)    
    
    
def plot_cluster_by_iteration(K, z, x, fig_name):
    fig, axis = plt.subplots(1, 3, figsize=(20,6))
    axis[0].scatter(x[:,1], x[:,0], c = z, alpha = 0.7, marker = ".", s = 10)
    axis[0].set_title("FL1.H vs FL2.H (ACDC, K = %i)"%(K))
    axis[1].scatter(x[:,2], x[:,1], c = z,alpha = 0.7, marker = ".", s = 10)
    axis[1].set_title("FL2.H vs FL3.H (ACDC, K = %i)"%(K))
    axis[2].scatter(x[:,3], x[:,2], c = z, alpha = 0.7, marker = ".", s = 10)
    axis[2].set_title("FL4.H vs FL3.H (ACDC, K = %i)"%(K))
    fig.tight_layout(h_pad=1)
    fig.savefig(fig_name+'-cluster-plot-K=%i.png'%(K))
    


def plot_loss_against_rho(kls, pis, K_true, K_select, rho_opt, maxK, lam, fig_name, log=True,legend=True,rho_min = 0, rho_max = 10):
    r = np.linspace(rho_min,rho_max,100)
    fig, axis = plt.subplots(1, 1, figsize=(6,3))
    Krange = np.arange(2,6)
    # Krange = np.arange(1,9)

    for pi, kl, K in zip(pis[Krange-1], kls[Krange-1], Krange):
    # for pi, kl, K in zip(pis, kls, np.arange(K_true-2,K_true+3)):
        y = np.sum(np.multiply(pi, np.maximum(kl-r[:,np.newaxis],0)),axis = 1) + lam * K
        # if K==K_true: 
        #     ypos = min(y) 
        #     print(y)
        axis.plot(r, y, lw=1.8,label = "K=%i"%K)
    # xpos = np.max(kls[K_true-1])+(np.max(kls[K_true-2])-np.max(kls[K_true-1]))/2
    axis.axvline(x=1.1616, c='black', ls='dashed',lw=1)
    # axis.legend(loc = 'upper center',fontsize = 20, bbox_to_anchor=(1.15, .9), fancybox=True, shadow=True)

    axis.set_xlabel(r'$\rho$',fontsize=20)
    axis.set_ylabel('Penalized loss',fontsize=20)
    axis.annotate('X', xy=(rho_opt+0.35, 30), color='red', fontsize=10, ha='center', va='center', weight='bold')
    axis.annotate(r'$K = %i$'%K_select, xy=(rho_opt+0.35, 0.6*32), fontsize=14, ha='center', va='center')
    axis.axvline(x=rho_opt, lw =2, linestyle='--',color='black')
    axis.annotate(r'$\rho \approx$%.2f'%rho_opt, xy=(rho_opt+0.35, max(y)+9500), color='black', fontsize=14, ha='center', va='center')

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
    
    

def plot_loss_against_rho_all(kls, pis, K_true, rho_opt, maxK, lam, fig_name, log=True,legend=True,rho_min = 0, rho_max = 10):
    r = np.linspace(rho_min,rho_max,100)
    fig, axis = plt.subplots(1, 1, figsize=(6,3))
    Krange = np.arange(1,9)

    for pi, kl, K in zip(pis[Krange-1], kls[Krange-1], Krange):
    # for pi, kl, K in zip(pis, kls, np.arange(K_true-2,K_true+3)):
        y = np.sum(np.multiply(pi, np.maximum(kl-r[:,np.newaxis],0)),axis = 1) + lam * K
        # if K==K_true: 
        #     ypos = min(y) 
        #     print(y)
        axis.plot(r, y, lw=1.8,label = "K=%i"%K)
    # xpos = np.max(kls[K_true-1])+(np.max(kls[K_true-2])-np.max(kls[K_true-1]))/2
    axis.axvline(x=1.1616, c='black', ls='dashed',lw=1)
    # axis.legend(loc = 'upper center',fontsize = 20, bbox_to_anchor=(1.15, .9), fancybox=True, shadow=True)

    axis.set_xlabel(r'$\rho$',fontsize=20)
    axis.set_ylabel('Penalized loss',fontsize=20)
    # axis.annotate('X', xy=(rho_opt, 20), color='red', fontsize=10, ha='center', va='center', weight='bold')
    # axis.annotate(r'$K = %i$'%2, xy=(rho_opt-0.2, 0.6*20), fontsize=13, ha='center', va='center')
    axis.axvline(x=rho_opt, lw =2, linestyle='--',color='black')
    axis.annotate(r'$\rho \approx$%.2f'%rho_opt, xy=(rho_opt+0.75, max(y)+10700), color='black', fontsize=14, ha='center', va='center')
    # axis.annotate(r'$\rho \approx$%.2f'%rho_opt, xy=(rho_opt+0.65, max(y)+50700), color='black', fontsize=14, ha='center', va='center')

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
    
    

    
def plot_fmeasure_against_K(z_sims, z_true, K_true, K_select, fig_name, legend):
    len_K = len(z_sims)
    f = np.zeros(len_K)
    for k in range(len_K):
        f[k] = F_measure(z_true,z_sims[k])

    fig, axis = plt.subplots(1, 1, figsize=(6,3))

    axis.plot(np.arange(1,len_K+1), f)
    axis.scatter([K_true, K_true-1], [f[K_true-1],f[K_true-2]], marker='*', label = r'$K_o$', s=80, color='r')

    axis.set_xlabel(r'$K$',fontsize=20)
    axis.set_ylabel('F-measure',fontsize=20)
    axis.xaxis.set_tick_params(labelsize=13)
    axis.yaxis.set_tick_params(labelsize=13)
    axis.vlines(x = K_select, label = r'$\hat{K}$', ymin=min(f), ymax=max(f), color='black', ls = '--')    
    # axis.vlines(x = K_true, label = 'Truth', ymin=min(f), ymax=max(f), color='r', ls = '--')
    sns.despine()

    if legend:
        axis.legend(prop={'size': 12})
        fig.savefig(fig_name+'-legend.pdf',bbox_inches='tight')    
    else:
        axis.legend('', frameon=False)
        fig.savefig(fig_name+'.pdf',bbox_inches='tight')  
    return f