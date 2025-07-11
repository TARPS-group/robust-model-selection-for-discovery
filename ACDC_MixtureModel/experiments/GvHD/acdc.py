#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import math
import collections
import numpy as np
from scipy.special import digamma
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.stats import multivariate_normal
from visualization import  plot_cluster_by_iteration

        
   
class FixACDC(object):


    def __init__(self, x, K, z_init = None):
        self.x = x
        self.K = K
        self.z_init = z_init
        freq = collections.Counter([tuple(y) for y in self.x])
        self.knn_lb = max(freq.values())
    

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
    
    
    def initialization(self, N_eff, reg_covar, scale, bias_correction):
        N, d = self.x.shape
        
        if self.z_init is None:
            km = KMeans(self.K)
            km.fit(self.x)
            mus = km.cluster_centers_            
            z = km.labels_
        else:
            z = self.z_init
            mus = [np.mean(self.x[z==k], axis = 0) for k in range(self.K)]
            
        Nk = np.unique(z, return_counts = True)[1]
        pis = Nk/N
 
        var = [np.cov(self.x[z==k].T) + reg_covar*np.identity(d) for k in range(self.K)]
        # z, mus, var, pis = self.reassign(z, pis, mus, var, N_eff, reg_covar) 
        K = len(mus)
        kl = np.zeros(K)
        for k in range(K):
            if not np.linalg.det(var[k]):
                print('init:check positive definite false', k, var[k],
                      'compk_obs', self.x[z==k], 'compk_size',len(self.x[z==k]),
                     'compk_mean', mus[k])
            kl[k] = self._kl_est(self.x[z==k], mus[k], var[k], scale, bias_correction)  
            
        return K, np.array(mus), np.array(var), np.array(pis), np.array(z)


    def update_params_labels(self, z, mus, var, pis, N_eff, reg_covar, scale, bias_correction, averaged_em):
        x = self.x    
        K = len(np.unique(z))
             
#         gmm = GMM(n_components = K, reg_covar = reg_covar)
#         gmm.fit(x)
#         mus = gmm.means_
#         var = gmm.covariances_
#         pis = gmm.weights_
#         z = gmm.predict(x)      

        N,d = x.shape
        if averaged_em:
            # print('running averaged EM')
            gamma = np.array([multivariate_normal.pdf(x, m, v) for m,v in zip(mus,var)])*np.array([multivariate_normal.pdf(xx, mus[zz], var[zz]) for xx,zz in zip(x,z)]) # K by N
        else:
            gamma = np.array([multivariate_normal.pdf(x, m, v) for m,v in zip(mus,var)]) # K by N
        
        gamma = np.multiply(gamma.T, pis)
        gamma = gamma/ gamma.sum(axis = 1)[:,np.newaxis]  # normalizing across components
        gamma_prime = gamma/ gamma.sum(axis = 0)[np.newaxis, :] # normalizing over observations    
        gamma_prime = gamma_prime.T
        
        Nk = gamma.sum(axis = 0)
        pis =  Nk if averaged_em else Nk/N
        mus = np.dot(gamma_prime, x)
        
        
        K = len(np.unique(z))
        var = []
        for k in range(K):
            gammak = np.sqrt(gamma_prime[k,:])    
            X_gamma = (x - mus[k]) * gammak[:, np.newaxis]
            var.append(X_gamma.T.dot(X_gamma) + np.identity(d)*reg_covar)
            
        # z, mus, var, pis = self.reassign(z, pis, mus, var, N_eff, reg_covar) 

        K = len(np.unique(z))
        kl = np.zeros(K)

        for k in range(K):

            kl[k] = self._kl_est(self.x[z==k], mus[k], var[k], scale, bias_correction) 
    
        z = self.random_sample(N, gamma)
        return z, mus, np.array(var), pis, kl
         
    

    def random_sample(self, N, weights):
        u = np.random.uniform(0, 1, N)
        c = u[:,np.newaxis] < np.cumsum(weights, axis=1)
        a = np.argwhere(np.cumsum(c, axis = 1) == 1)
        return a[:,1]  

    
    # def reordering(self, z_sm, reg_covar):
    #     x, K_sm = self.x, len(np.unique(z_sm))
    #     labels = np.unique(z_sm)
    #     N,d = x.shape

    #     pis = np.zeros(K_sm)
    #     mus = []
    #     var = []
    #     for i in range(K_sm):
    #         z_sm[z_sm == labels[i]] = i
    #         mus.append(np.mean(x[z_sm == i], axis = 0))
    #         var.append(np.cov(x[z_sm == i].T)+np.identity(d)*reg_covar)          
    #         pis[i] = np.mean(z_sm == i)
    #     return z_sm, np.array(mus), np.array(var), pis    
 

   
    # def reassign(self, z, pis, mus, var, N_eff,reg_covar):
    #     Nk = np.unique(z, return_counts=True)[1]
    #     removed = Nk < N_eff
    #     x = self.x
    #     idx_removed = np.where(removed)[0]
    #     idx = np.where(~removed)[0]
    #     for i in range(len(z)):
    #          if z[i] in idx_removed:    
    #             marginal_xs = np.multiply(pis, [multivariate_normal.pdf(x[i], m, v) for m,v in zip(mus,var)])
    #             marginal_xs /= np.sum(marginal_xs)
    #             pos = np.delete(marginal_xs, idx_removed)
    #             pos /= np.sum(pos)
    #             z[i] = random.choices(idx, pos)[0]
    #     z, mus, var, pis = self.reordering(z,reg_covar)
    #     return z, mus, var, pis



    def fit(self, N_eff, iterations, reg_covar, scale, bias_correction, averaged_em):

        K, mus, var, pis, z = self.initialization(N_eff, reg_covar, scale, bias_correction)
        kl = np.zeros(K)


        for it in range(1, iterations): 
            
            # z, mus, var, pis = self.reassign(z, pis, mus, var, N_eff,reg_covar) 
            z, mus, var, pis, kl = self.update_params_labels(z, mus, var, pis, N_eff, reg_covar, scale, bias_correction, averaged_em)
        loss = sum(np.multiply(pis,kl))
        return z, mus, var, pis, kl, loss

    

class ACDC(object):
    
# Start with K = 1,..., max_K
#     for each k = 1, ..., K, split k-th component into two
#      - compute original loss_i = N_i * KL(pi | f_theta_i), Loss = sum_i loss_i, i = 1, ..., K+1
#      - pick k-th component with the lowest loss 
#          -> store its assignment z as the initial for next iteration
#          -> store the parameters mu, sigma, pi?
#          -> store kl_i = KL(pi | f_theta_i)
#      - compute modified loss with penalization_i = sum_i pi_i * max(kl_i-rho, 0) + lam * (K+1) with rho as grid 


    def __init__(self, x, noisy=False):
        self.x = x
        if noisy:
            N,d = self.x.shape
            self.x_noisy = x + multivariate_normal.rvs([0]*d, np.identity(d), N)
        

    def _split_components(self, z, c_j):
        z_sm = z.copy()
        z_km = z_sm[z_sm == c_j]
        km = KMeans(2)
        km.fit(self.x[z == c_j])
        z_km[km.labels_ == 0] = np.max(z) + 1
        z_km[km.labels_ == 1] = c_j
        z_sm[z_sm == c_j] = z_km
        return z_sm

    
    def fit(self, N_eff, iter_update, fig_name, scale, bias_correction = True, reg_covar = 0.01, eps = 1e-1, maxK = 10, averaged_em=False, seed=10):  
        split_kth_sims = []
        kl_sims = []
        mus_sims = []
        sds_sims = []
        pis_sims = []
        loss_sims = []
        z_sims = []
        for K in range(1,maxK):
            print('currK',K)
            if K == 1:
                probcare_Kinit = FixACDC(x = self.x_noisy, K = K)
                min_z, min_mus, min_sds, min_pis, min_kl, min_loss = probcare_Kinit.fit(N_eff, iter_update, reg_covar, scale, bias_correction, averaged_em)
                min_k = []
            else:
                temp = min_z[:]
                min_loss = float('inf')
                for k in range(K-1):
                    z_sm = self._split_components(temp, k)
                    probcare_Kinit = FixACDC(x = self.x_noisy, K = K, z_init = z_sm)
                    z, mus, sds, pis, kl, loss = probcare_Kinit.fit(N_eff, iter_update, reg_covar, scale, bias_correction, averaged_em)
                    print('split %i-th component gives loss %.4f'%(k,loss))
                    if loss < min_loss:
                        min_k, min_z, min_mus, min_sds, min_pis, min_kl, min_loss = k, z, mus, sds, pis, kl, loss
                print('split %i-th component with KL = %.4f'%(min_k,kl_sims[-1][min_k]), 'prev kls', kl_sims[-1])
            
            z_sims.append(min_z)
            mus_sims.append(min_mus)
            sds_sims.append(min_sds)
            pis_sims.append(min_pis)
            kl_sims.append(min_kl)
            loss_sims.append(min_loss)
            split_kth_sims.append(min_k)
            
            plot_cluster_by_iteration(K, min_z, self.x, fig_name)
        
                        
        return z_sims,mus_sims,sds_sims,pis_sims,kl_sims,loss_sims,split_kth_sims

    



