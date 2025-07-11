#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import poisson
from visualization import  plot_cluster_by_iteration

class PoissonMixtureModel():
    def __init__(self):
        pass
    
    def log_likelihood(self, data, weights, lambdas):
        num_components = len(weights)
        mix_model = np.sum([weights[i] * poisson.pmf(data, lambdas[i]) for i in range(num_components)], axis=0)
        return np.sum(np.log(mix_model))
    
    def fit(self, data, num_components, num_iterations=100):
        n_samples = len(data)
        weights = np.ones(num_components) / num_components
        lambdas = np.random.uniform(low=data.min(), high=data.max(), size=num_components)
        
        for _ in range(num_iterations):
            # E-step
            responsibilities = np.zeros((n_samples, num_components))
            for i in range(num_components):
                responsibilities[:, i] = weights[i] * poisson.pmf(data, lambdas[i])
            responsibilities /= responsibilities.sum(axis=1)[:, np.newaxis]
            
            # M-step
            for i in range(num_components):
                weights[i] = responsibilities[:, i].sum() / n_samples
                lambdas[i] = np.sum(responsibilities[:, i] * data) / np.sum(responsibilities[:, i])
        self.weights=weights
        self.lambdas=lambdas
        
    def predict(self, data):
        num_components = len(self.weights)
        n_samples = len(data)
        responsibilities = np.zeros((n_samples, num_components))
    
        # Calculate responsibilities for each component
        for i in range(num_components):
            responsibilities[:, i] = self.weights[i] * poisson.pmf(data, self.lambdas[i])
        
        # Assign labels based on the component with the highest probability
        labels = np.argmax(responsibilities, axis=1)
        return labels
    
    def means_(self):
        return self.lambdas
    
    def weights_(self):
        return self.weights
        

class ACDC():
    

    def __init__(self, x):
        self.x = x

    def _kl_plugin(self, sk, lam_k):
        kl_k = 0
        N_is = np.unique(sk, return_counts=True)[1]
        x_is = np.unique(sk)
        for it in range(len(x_is)):
            temp =  N_is[it]/ len(sk)
            kl_k += temp * (np.log(temp)- poisson.logpmf(x_is[it], lam_k))  
        return kl_k 
    
    

    def fit(self, maxK, fig_name):
        print('Running ACDC...')
        z_sims = []
        mus_sims = []
        pis_sims = []
        loss_sims = []
        kl_sims = []
        for K in range(1,maxK):
            print('iteration K=%i'%K)
            pois_mod = PoissonMixtureModel()
            pois_mod.fit(data=self.x, num_components=K)
            z = pois_mod.predict(self.x)
            pis = pois_mod.weights_()
            mus = pois_mod.means_()
            divergence_comp = np.zeros(K)
            for k in range(K):
                divergence_comp[k]=self._kl_plugin(self.x[z==k], mus[k])
                print('divergence for %ith component:'%(k+1), divergence_comp[k])
            loss = sum(np.multiply(pis, divergence_comp))

            kl_sims.append(divergence_comp)
            loss_sims.append(loss)
            z_sims.append(z)
            mus_sims.append(mus)
            pis_sims.append(pis)
            plot_cluster_by_iteration(K, z, self.x, fig_name)

        return z_sims, mus_sims, pis_sims, loss_sims, kl_sims



