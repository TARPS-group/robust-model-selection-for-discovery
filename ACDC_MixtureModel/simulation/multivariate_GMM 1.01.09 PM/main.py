#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from matplotlib import pyplot as plt
from stare import STARE
from data import DSSkewNormalMixture
from sklearn.mixture import GaussianMixture as GMM
from utils import Timer, create_folder_if_not_exist
from visualizations import (plot_loss_against_rho,
                            plot_true_cluster,
                            plot_cluster)

dir_path = os.getcwd()
create_folder_if_not_exist(dir_path+'/figures')
create_folder_if_not_exist(dir_path+'/results')


if __name__ == "__main__":
    K_true = 3
    N = 10000
    D = 50
    sigma = 0.6 # hyperparameter to control the correlation through covariance matrix
    
    mu1, mu2, mu3 = np.random.normal(1, 0.5, D+1), np.random.normal(6, 0.5, D+1), np.random.normal(10, 0.5, D+1)
    means = np.concatenate((mu1, mu2, mu3)).reshape(3,D+1)
    pmix = np.array([0.70, 0.20, 0.10])
    a1, a2, a3 = np.random.normal(2, 1, D), np.random.normal(4, 1, D), np.random.normal(10, 1, D)
    a = np.concatenate((a1, a2, a3)).reshape(3,D)
    
    simulator = DSSkewNormalMixture(K_true, D, means, a, pmix, sigma)
    # simulator.cov
   
    x, z = simulator.sample(N, seed=111)
    plot_true_cluster(10, K_true, x[:, 0:10], z, fig_name = 'figures/multiGMM-true-cluster-K-%.f'%(K_true))

    maxK = 5
    

    # STARE Inference
    with Timer('STARE'):
        N_eff = int(np.sqrt(N))+1
        stare = STARE(x)
        z_sims, mus_sims, sds_sims, pis_sims, loss_sims, kl_sims = stare.fit(maxK=maxK, scale = 1/2, bias_correction=True)
      
    np.save('results/stare-n=%i-d=%i-z_sims'%(N,D), z_sims)
    np.save('results/stare-n=%i-d=%i-pis_sims'%(N,D), pis_sims)
    np.save('results/stare-n=%i-d=%i-kl_sims'%(N,D), kl_sims)
    np.save('results/stare-n=%i-d=%i-z_true'%(N,D), z)

    pis_sims = np.load('results/stare-n=%i-d=%i-pis_sims.npy'%(N,D), allow_pickle=True)
    kl_sims = np.load('results/stare-n=%i-d=%i-kl_sims.npy'%(N,D), allow_pickle=True)
    
    plot_loss_against_rho(kl_sims, pis_sims, K_true, maxK = 5, lam = 0.01, rho_min = 0, rho_max = 1, legend=False,log=True,fig_name = 'figures/multiGMM-penloss-rho-n=%i-d=%i'%(N,D))
    plot_loss_against_rho(kl_sims, pis_sims, K_true, maxK = 5, lam = 0.01, rho_min = 0, rho_max = 1, legend=True,log=True,fig_name = 'figures/multiGMM-penloss-rho-n=%i-d=%i'%(N,D))

    plot_cluster(10, z_sims[2], x[:, 0:D], title='STARE', fig_name = 'figures/multiGMM-STARE-cluster-n=%i-d=%i'%(N,D))


    # selected cluster plot comparison
    fig, axis = plt.subplots(1,2, figsize=(6,3), sharex = 'col', sharey = True)
    for i in [2,3]:
        axis[i-2].scatter(x[:,i], x[:,i+1], c = z, alpha = 0.7, marker = ".", s = 10)
        axis[i-2].xaxis.set_tick_params(labelsize=20)
        axis[i-2].yaxis.set_tick_params(labelsize=20)
        # axis[i-2].set_title("Dimension %.f vs Dimension %.f"%(i+1, i+2),fontsize=26)
        axis[i-2].set_xlabel('$x_{%i}$'%(i+1), fontsize=24)
        axis[i-2].set_ylabel('$x_{%i}$'%(i+2), fontsize=24)
       # axis[1,i].set_title("Dimension%.f vs Dimension%.f (Structurally-aware selection, \hat{K} = %i)"%(i+1, (i+2), K_true), fontsize=18)

    fig.tight_layout(h_pad=1.5)
    fig.savefig('figures/selected-cluster-plot-comparison-K=%i.png'%(K_true))


    # BIC criterion
    n_components = np.arange(1, maxK)
    models = [GMM(n, covariance_type='full', random_state=0).fit(x) for n in n_components]
    bic = [m.bic(x) for m in models]
    bic_n = n_components[np.where(bic == np.min(bic))]
    print('BIC criterion chooses K=%i'%bic_n)
    gmm = GMM(n_components=bic_n[0]).fit(x)
    z_em = gmm.predict(x)
    plot_cluster(10, z_em, x[:, 0:D], title='BIC', fig_name = 'figures/multiGMM-BIC-cluster-n=%i-d=%i'%(N,D))

    
    
    
    