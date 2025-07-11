#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
from scipy.stats import nbinom, dirichlet


class DSNegBinMixture(object):
    def __init__(self, K, n = None, p = None, pmix = None):      
        if pmix is None:
            alpha = random.sample(range(1,100), K)
            pmix = dirichlet.rvs(alpha, size = 1).reshape(-1)     
            
        if n is None:
            n = random.sample(range(1,100), K)
            
        if p is None:
            p = [0.5]*K
        
        if len(n) != len(p):
            raise ValueError('Number of components in n and p do not match.')
        
        if np.max(p) - 1 > 1e-8:
            raise ValueError('Success probablity is not below 1.')
            
        if np.abs(np.sum(pmix) - 1) > 1e-8:
            raise ValueError('Mixture weights do not sum to 1.')
                    
        self.K = K
        self.pmix = pmix
        self.n = n
        self.p = p
        self.z_true = None
    
    def ncomp(self):
        return self.K

    def mean(self):
        return self.n * self.p/(1 - self.p)
    
    def true_cluster(self):
        return self.z_true
    
    def logpmf(self, x, k):
        return nbinom.logpmf(x, self.n[k], self.p[k])
    
    def pmf(self, x, k):
        return self.pmix[k] * np.exp(self.logpmf(x, k))    
    
    def sample(self, N, seed=111): 
        x = np.zeros(N)
        z = np.zeros(N)
        for n in range(N):
            k = random.choices(list(range(self.K)), self.pmix)[0]
            x[n] = np.random.negative_binomial(self.n[k], self.p[k])
            z[n] = k
        self.z_true = z
        return x    

