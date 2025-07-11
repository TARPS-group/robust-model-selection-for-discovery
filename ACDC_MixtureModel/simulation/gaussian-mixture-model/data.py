#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains various data generators for simulation study.
"""
import random
import numpy as np
from scipy.stats import skewnorm


class DSSkewMixture(object):

    def __init__(self, K, means, sds, a, pmix=None):
        
        K = len(means)
        
        if K != len(sds):
            raise ValueError('Number of components in means and variances do not match.')
        
        assert len(a) == K
        
        if pmix is None:
            pmix = np.ones(K)/float(K)

        if np.abs(np.sum(pmix) - 1) > 1e-8:
            raise ValueError('Mixture weights do not sum to 1.')
            
        self.K = K        
        self.a = a  # skewness parameter
        self.pmix = pmix
        self.means = means
        self.sds = sds
        self.z_true = None
        self.sample_size = None
        
    def ncomp(self):
        return self.K
   
    def sample_size(self):
        return self.sample_size
    
    
    def true_cluster(self):
        return self.z_true
    
    def mean(self):
        return self.means    
    
    def pdf(self, x, k):
        return self.pmix[k] * skewnorm.pdf(x, self.a[k], self.means[k], self.sds[k]) 
    
    def logpdf(self, x, k):
        return self.pmix[k] * skewnorm.logpdf(x, self.a[k], self.means[k], self.sds[k]) 
    
    def sample(self, N, seed=111):
        means = self.means
        sds = self.sds
        self.sample_size = N
        x = np.zeros(N)
        z = np.zeros(N)
        for n in range(N):
            k = random.choices(range(self.K), self.pmix)[0]
            x[n] = skewnorm.rvs(self.a[k], loc = means[k], scale = sds[k], size = 1)        
            z[n] = k
        self.z_true = z
        return x      
    

