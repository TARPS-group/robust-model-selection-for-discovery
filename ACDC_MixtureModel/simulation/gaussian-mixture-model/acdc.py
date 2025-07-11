#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the main ACDC algorithm
"""
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import multivariate_normal
from scipy.special import digamma
from sklearn.mixture import GaussianMixture as GMM
from visualization import plot_cluster_by_iteration



class ACDC():
    '''    
     Start with K = 1,..., max_K
         for each k = 1, ..., K, split k-th component into two
          - compute original loss_i = N_i * KL(pi | f_theta_i), Loss = sum_i loss_i, i = 1, ..., K+1
          - pick k-th component with the lowest loss 
              -> store its assignment z as the initial for next iteration
              -> store the parameters mu, sigma, pi?
              -> store kl_i = KL(pi | f_theta_i)
          - compute modified loss with penalization_i = sum_i pi_i * max(kl_i-rho, 0) + lam * (K+1) with rho as grid 
    '''

    def __init__(self, x):
        self.x = x
        
        
    
    def _kl_est(self, sk, muk, vark, scale, bias_correction):
        d = self.x.shape[1]
        kl_k = 0
        N_k = len(sk)
        knn_min = np.unique(sk, return_counts = True, axis = 0)[1][0]+1
        if scale is not None:
            knn = max(int(N_k ** scale), knn_min)
            # print('knn',int(N_k ** scale), 'min_rep', knn_min-1)
        else:
            knn = int(math.sqrt(len(self.x)))
        if bias_correction:
            bias_correction = digamma(knn) - np.log(knn)
        else:
            bias_correction = 0
        knn_mod = NearestNeighbors(n_neighbors = knn+1)
        knn_mod.fit(sk)
        vis, ind = knn_mod.kneighbors(sk)
        for i in range(N_k):
            vi = vis[i][knn]         
            volume = math.pi**(d/2)/math.gamma(d/2 + 1)*vi**d
            kl_k += np.log(knn) - np.log(N_k - 1) - np.log(volume) - multivariate_normal.logpdf(sk[i], muk, vark)        
        return kl_k/N_k + bias_correction        
    
    def fit(self, maxK, fig_name, scale=1/2, bias_correction=True):
        print('Running ACDC...')
        z_sims = []
        mus_sims = []
        sds_sims = []
        pis_sims = []
        loss_sims = []
        kl_sims = []
        for K in range(1,maxK):
            print('iteration K=%i'%K)
            gm = GMM(n_components=K, random_state=0, covariance_type='diag').fit(self.x)
            z = gm.predict(self.x)
            pis = gm.weights_
            mus = gm.means_
            var = gm.covariances_
            divergence_comp = np.zeros(K)
            for k in range(K):
                divergence_comp[k]=self._kl_est(self.x[z==k], mus[k], var[k], scale, bias_correction)
                print('divergence for %ith component:'%(k+1), divergence_comp[k])
            loss = sum(np.multiply(pis, divergence_comp))

            kl_sims.append(divergence_comp)
            loss_sims.append(loss)
            z_sims.append(z)
            mus_sims.append(mus)
            sds_sims.append(var) 
            pis_sims.append(pis)
            plot_cluster_by_iteration(K, z, self.x, fig_name)
        return z_sims, mus_sims, sds_sims, pis_sims, loss_sims, kl_sims

    



