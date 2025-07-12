#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal

class DSSkewNormalMixture():

    def __init__(self, K, D, means, a, pmix, sigma):
        '''
        sigma: parameter to control the correlation; 
        '''
        
        K = len(means)
        
        assert len(a) == K
        
        if np.abs(np.sum(pmix) - 1) > 1e-8:
            raise ValueError('Mixture weights do not sum to 1.')
            
        self.K = K     
        self.D = D
        self.means = means
      
        self.cov = self.generate_cov(sigma)
  
        self.a = a  # skewness parameter
        self.pmix = pmix
        
    
    def generate_cov(self, sigma):
        A = np.arange(self.D)
        corr = np.exp(-(A[:,np.newaxis] - A[np.newaxis,:])**2 / sigma**2)
        I = np.eye(self.D, self.D)
        cov =  I@corr@I 
        return cov

    def _generate_data(self, mean, a, size = 1, seed = 111):
        alphaTcovalpha = np.transpose(a)@(self.cov)@a
        delta = (1 / np.sqrt(1 + alphaTcovalpha)) * self.cov @a
        cov_star = np.block([[np.ones(1),delta],[delta[:, None], self.cov]])
        y        = multivariate_normal(mean, cov_star).rvs(size)
        y0, y1   = y[0], y[1:]
        inds     = y0 <= 0
        y1[inds] = -1 * y1[inds]
        return y1
    
    def sample(self, N, seed):
        x = np.zeros((N, self.D))
        z = np.zeros(N)
        for i in range(N):
            label = np.random.choice(range(self.K), p=self.pmix)
            x[i] = self._generate_data(self.means[label], self.a[label], 1, seed) 
            z[i] = label 
        self.z_true = z
        return x, z  
    
    def ncomp(self):
        return self.K
    
    def mean(self):
        return self.means   

    def cov(self):
        return self.cov

    def true_labels(self):
        return self.z_true