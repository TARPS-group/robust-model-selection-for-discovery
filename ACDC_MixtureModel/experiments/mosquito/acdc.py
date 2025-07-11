#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import bernoulli

class multivariate_bernoulli(object):
    def __init__(self):
        pass
    
    def logpmf(self, x, p):
        '''
        Assuming independent between x_i and x_j
        x: 1 by D vector 
        p: 1 by D vector, 0 <= pi <= 1
        '''
        return sum([bernoulli.logpmf(xi, pi) for xi, pi in zip(x,p)])
    
    def pmf(self, x, p):
        return np.exp(self.logpmf(x,p))
    
# globally call multivariate_bernoulli class
multivariate_bernoulli = multivariate_bernoulli()
     
   
class FixACDC(object):


    def __init__(self, x, K, z_init = None):
        self.x = x
        self.K = K
        self.z_init = z_init

    

    def _kl_plugin(self, sk, lam_k):
        kl_k = 0
        Nk = len(sk)
        lam_k[lam_k<=0]=1e-15  # tackle numerical overflows
        lam_k[lam_k>=1]=1-1e-15
        uni, cnt = np.unique(sk, axis=0, return_counts = True)
        for i in range(len(cnt)):
            kl_k += cnt[i]/Nk*(np.log(cnt[i]/Nk) - multivariate_bernoulli.logpmf(uni[i], lam_k))
            if multivariate_bernoulli.logpmf(uni[i], lam_k) < -1e5:
                print('xi',uni[i],'lamk',lam_k)
        return kl_k       
    
    
    def initialization(self):
        N = len(self.x)
        km = KMeans(self.K)
        km.fit(self.x)
        mus = km.cluster_centers_


        z_km = km.labels_
        
        if self.z_init is None:
            z = z_km
        else:
            z = self.z_init
            
        Nk = np.unique(z, return_counts = True)[1]
        pis = Nk/N
#         z, mus, pis = self.reassign(z, pis, mus, N_eff) 
        K = len(mus)
        kl = np.zeros(K)
 
        for k in range(K):
            kl[k] = self._kl_plugin(self.x[z==k], mus[k])
            print('%i-th kl'%k,kl[k])
            
        return K, np.array(mus), np.array(pis), np.array(z)


 
    def update_params_labels(self, z, mu, pi):
        x = self.x          
        N = len(x)
        
        gamma = np.array([[multivariate_bernoulli.pmf(xi, m) for xi in x] for m in mu]) # K by N
        #print('gamma dim', gamma.shape, 'pi dim', len(pi))
        gamma = np.multiply(gamma.T, pi) # N by K
        gamma = gamma/ gamma.sum(axis = 1)[:,np.newaxis]  # normalizing across components -> E(z_nk)
        gamma_prime = gamma/ gamma.sum(axis = 0)[np.newaxis, :] # normalizing over observations
        Nk = gamma.sum(axis = 0)  


        mu = gamma_prime.T.dot(x) # K by D
                
        K = len(Nk)
        kl = np.zeros(K)
        
        for k in range(K):
            pi[k] = Nk[k]/N
            kl[k] = self._kl_plugin(x[z==k], mu[k])

        z = self.random_sample(N, gamma)
        return z, mu, pi, kl  
         
    

    def random_sample(self, N, weights):
        u = np.random.uniform(0, 1, N)
        c = u[:,np.newaxis] < np.cumsum(weights, axis=1)
        a = np.argwhere(np.cumsum(c, axis = 1) == 1)
        return a[:,1]  



    def fit(self, iterations):

        K, mus, pis, z = self.initialization()
        kl = np.zeros(K)


        for it in range(1, iterations): 
            
            z, mus, pis, kl = self.update_params_labels(z, mus, pis)
        loss = sum(np.multiply(pis,kl))
        return z, mus, pis, kl, loss

    

class ACDC(object):
    
# Start with K = 1,..., max_K
#     for each k = 1, ..., K, split k-th component into two
#      - compute original loss_i = N_i * KL(pi | f_theta_i), Loss = sum_i loss_i, i = 1, ..., K+1
#      - pick k-th component with the lowest loss 
#          -> store its assignment z as the initial for next iteration
#          -> store the parameters mu, sigma, pi?
#          -> store kl_i = KL(pi | f_theta_i)
#      - compute modified loss with penalization_i = sum_i pi_i * max(kl_i-rho, 0) + lam * (K+1) with rho as grid 


 
    def __init__(self, x):
        self.x = x

    def _split_components(self, z, c_j):
        z_sm = z.copy()
        z_km = z_sm[z_sm == c_j]
        km = KMeans(2)
        km.fit(self.x[z == c_j])
        z_km[km.labels_ == 0] = np.max(z) + 1
        z_km[km.labels_ == 1] = c_j
        z_sm[z_sm == c_j] = z_km
        return z_sm

    
    def fit(self, iter_update, eps = 1e-1, maxK = 10, seed=10):  
        split_kth_sims = []
        kl_sims = []
        mus_sims = []
        pis_sims = []
        loss_sims = []
        z_sims = []
#         fig, axis = plt.subplots(1, 1, figsize=(10,6))
        for K in range(1,maxK):

            print('currK',K)
            if K == 1:
                probcare_Kinit = FixACDC(x = self.x, K = K)
                min_z, min_mus, min_pis, min_kl, min_loss = probcare_Kinit.fit(iter_update)
                min_k = []
            else:
                temp = min_z[:]
                min_loss = float('inf')
                for k in range(K-1):
                    z_sm = self._split_components(temp, k)
                    probcare_Kinit = FixACDC(x = self.x, K = K, z_init = z_sm)
                    z, mus, pis, kl, loss = probcare_Kinit.fit(iter_update)
                    print('split %i-th component gives loss %.4f'%(k,loss))
                    if loss < min_loss:
                        min_k, min_z, min_mus,min_pis, min_kl, min_loss = k, z, mus, pis, kl, loss
                print('split %i-th component with KL = %.4f'%(min_k,kl_sims[-1][min_k]), 'prev kls', kl_sims[-1])
            
            z_sims.append(min_z)
            mus_sims.append(min_mus)
            pis_sims.append(min_pis)
            kl_sims.append(min_kl)
            loss_sims.append(min_loss)
            split_kth_sims.append(min_k)
            
#             plot_cluster_by_iteration(K, min_z, self.x, fig_name)
            
#         loss_sims.append(losses)
#         z_sims.append(zs)

#         axis.plot(rhos, losses, label = "K=%i"%K)
#         axis.legend(fontsize = 14)
#         axis.set_xlabel(r'$\rho$',fontsize=13)
#         axis.set_ylabel('Loss',fontsize=13)
#         fig.savefig(fig_name+'-loss-plot.png')
                        
        return z_sims,mus_sims,pis_sims,kl_sims,loss_sims,split_kth_sims



