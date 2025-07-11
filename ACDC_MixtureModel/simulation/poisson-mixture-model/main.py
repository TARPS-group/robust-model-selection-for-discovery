#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson, uniform
from utils import Timer, create_folder_if_not_exist
from data import DSNegBinMixture
from acdc import ACDC
from visualization import (plot_true_density, 
                           plot_true_cluster,
                           plot_fmeasure_against_K,
                           plot_loss_against_rho)



dir_path = os.getcwd()
create_folder_if_not_exist(dir_path+'/figures')    
create_folder_if_not_exist(dir_path+'/results')                  
              

###################
##### Set up ######
###################

K_t = 3
N = 20000
n = np.array([55, 75, 100])
pmix= np.array([0.3, 0.3, 0.4])
p = np.array([0.5, 0.3, 0.5])

data = DSNegBinMixture(K = K_t, n = n, p = p, pmix = pmix)
x = data.sample(N = N)
z_true = data.true_cluster()

plot_true_cluster(x, z_true, figname = "figures/cluster-plot-true-n=%i-K0=%i.png"%(N,K_t))
plot_true_density(data, figname = "figures/density-plot-true-K0=%i.pdf"%K_t, xmax = 250)

###########################
##### Run ACDC algo ######
###########################

with Timer('Run ACDC'):
    N = len(z_true)
    maxK = 9
    acdc = ACDC(x)
    z_sims, mus_sims, pis_sims, loss_sims, kl_sims= acdc.fit(fig_name = 'figures/acdc-N=%i-trueK=%i'%(N,K_t), maxK = maxK)
    
np.save('results/acdc-n=%i-k=%i-z_sims'%(N,K_t), z_sims)
np.save('results/acdc-n=%i-k=%i-pis_sims'%(N,K_t), pis_sims)
np.save('results/acdc-n=%i-k=%i-kl_sims'%(N,K_t), kl_sims)
np.save('results/acdc-n=%i-k=%i-z_true'%(N,K_t), z_true)    
#######################################
##### Visualization and analysis ######
#######################################
z_sims = np.load('results/acdc-n=%i-k=%i-z_sims.npy'%(N,K_t), allow_pickle=True)
z_true = np.load('results/acdc-n=%i-k=%i-z_true.npy'%(N,K_t), allow_pickle=True)
plot_fmeasure_against_K(z_sims, z_true, K_t, fig_name = 'figures/fmeasure-N=%i-trueK=%i.pdf'%(N,K_t))


pis_sims = np.load('results/acdc-n=%i-k=%i-pis_sims.npy'%(N,K_t), allow_pickle=True)
kl_sims = np.load('results/acdc-n=%i-k=%i-kl_sims.npy'%(N,K_t), allow_pickle=True)
plot_loss_against_rho(kl_sims, pis_sims, K_t, maxK = 5, lam = .01, log=True, legend=True,rho_min = 0, rho_max = 15, fig_name = 'figures/penloss-plot-N=%i-trueK=%i'%(N,K_t))
plot_loss_against_rho(kl_sims, pis_sims, K_t, maxK = 5, lam = .01, log=True, legend=False,rho_min = 0, rho_max = 15, fig_name = 'figures/penloss-plot-N=%i-trueK=%i'%(N,K_t))

# cluster plot
y = uniform.rvs(size = len(x))
fig, axis = plt.subplots(2, 1, figsize=(10,3))
axis[0].scatter(x, y, c= z_true, marker = ".")
#axis[0].set_title( 'Truth, $K_o = %.f$'%K_t, fontsize= 16)
axis[0].set_yticks([])
axis[0].xaxis.set_tick_params(labelsize=16)
axis[1].scatter(x, y, c= z_sims[K_t-1], marker = ".")
#axis[1].set_title( 'Structurally-aware Criterion, $\hat{K} = %.f$'%K_t, fontsize= 16)
axis[1].set_yticks([])
axis[1].xaxis.set_tick_params(labelsize=16)
fig.tight_layout(h_pad=1)
fig.savefig('figures/acdc-true-cluster-plot-notitle.png')

# pmf comparison
xmin, xmax = -10,300
fig, axis = plt.subplots(1, 1, figsize=(6,3))

grid = np.arange(xmin, xmax, 4)
truth = np.zeros(len(grid))  


for k in range(K_t):
    truth += data.pmf(grid, k)   
axis.stem(grid, truth, linefmt='r-', markerfmt='ro', basefmt='r-', use_line_collection = True)

acdc = np.zeros(len(grid))
for k in range(K_t):
    acdc = pis_sims[K_t-1][k] * poisson.pmf(grid, mus_sims[K_t-1][k])
    # markerline, stemlines, baseline = axis.stem(grid, acdc, '-')
    # axis.set_setp(baseline, 'linewidth', 2, 'markersize',1)

    axis.stem(grid, acdc, linefmt='-', markerfmt='o', basefmt='-', use_line_collection = True)     

axis.legend(["True distribution", "ACDC"], prop = dict(size=18))  
#axis.legend('', frameon=False)
axis.set_xlabel("X", size = 20)
axis.set_ylabel("Pmf", size = 20)
axis.xaxis.set_tick_params(labelsize=16)
axis.yaxis.set_tick_params(labelsize=16)
sns.despine()
fig.tight_layout(h_pad=1)
fig.savefig('figures/negbin-acdc-pmfs-n=%i-Kt=%i-legend.pdf'%(N,K_t))



# Histograms 
import random
x_model = np.zeros(N)
for n in range(N):
    k = random.choices(list(range(K_t)), pis_sims[K_t-1])[0]
    x_model[n] = np.random.poisson(mus_sims[K_t-1][k])

fig, axis = plt.subplots(1, 1, figsize=(6,3))
sns.histplot(x, ax=axis, bins=30, kde=False,  label='Data', color='dodgerblue', stat="density", edgecolor='white' )
sns.histplot(x_model,ax=axis, bins=30, kde=False, label='Model',color='darkorange', stat="density", edgecolor='white')
axis.set_xlabel('x', fontsize = 20)
axis.set_ylabel('Frequency', fontsize = 20)
axis.xaxis.set_tick_params(labelsize=13)
axis.yaxis.set_tick_params(labelsize=13)
axis.legend(prop={'size': 12})
sns.despine()
fig.savefig('figures/pois-hist-legend.pdf',bbox_inches='tight')


