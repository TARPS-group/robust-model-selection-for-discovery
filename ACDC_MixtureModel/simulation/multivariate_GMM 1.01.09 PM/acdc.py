#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture as GMM


class STARE():
    def __init__(self, x):
        self.x = x
    
    def knn_estimate(self, sk, muk, vark, scale, bias_correction):    
         N_k = len(sk)
         d = self.x.shape[1]
                
         ### KL by coordinate ####
         math_calc = 0
         for cord in range(d):
             kl_k = 0
             sk_d = np.array(sk[:, cord])
             knn_min = np.unique(sk_d, return_counts = True, axis = 0)[1][0]+1
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
             knn_mod.fit(sk_d.reshape(-1,1))
             vis, ind = knn_mod.kneighbors(sk_d.reshape(-1,1))
             for i in range(N_k):
                 vi = vis[i][knn]         
                 volume = math.pi**(1/2)/math.gamma(1/2 + 1)*vi
                 kl_k += np.log(knn) - np.log(N_k - 1) - np.log(volume) - multivariate_normal.logpdf(sk_d[i], muk[cord], vark[cord])        
             math_calc += kl_k/N_k + bias_correction        
         return math_calc/d        
 
    
    def fit(self, maxK, scale=1/2, bias_correction=True):
        print('Running STARE...')
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
                divergence_comp[k]=self.knn_estimate(self.x[z==k], mus[k], var[k], scale, bias_correction)
                print('divergence for %ith component:'%(k+1), divergence_comp[k])
            loss = sum(np.multiply(pis, divergence_comp))

            kl_sims.append(divergence_comp)
            loss_sims.append(loss)
            z_sims.append(z)
            mus_sims.append(mus)
            sds_sims.append(var) 
            pis_sims.append(pis)
        return z_sims, mus_sims, sds_sims, pis_sims, loss_sims, kl_sims



    
    