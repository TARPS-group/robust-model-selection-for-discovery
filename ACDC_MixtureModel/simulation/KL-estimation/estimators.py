#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from sklearn.neighbors import NearestNeighbors,KernelDensity
from scipy.stats import multivariate_normal, poisson,entropy

def knn_estimator(x, mu, sig, knn):
    d = len(mu)
    kl = 0
    N = len(x)
    knn_mod = NearestNeighbors(n_neighbors = knn+1)
    knn_mod.fit(x)
    vis, ind = knn_mod.kneighbors(x)
    for i in range(N):
        vi = vis[i][knn]         
        volume = math.pi**(d/2)/math.gamma(d/2 + 1)*vi**d
        kl += np.log(knn) - np.log(N - 1) - np.log(volume) - multivariate_normal.logpdf(x[i], mu, sig)        
    return kl/N       

def scott_bw(n,d):
    return n**(-1/(d+4))

def kde_estimator(x, mu, sig, bandwidth = 'scott', kernel = 'gaussian'):
    N, d = len(x), len(mu)
    kl = 0
    if bandwidth == 'scott':
        bw = scott_bw(N, d)
    kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(x)    
    log_density = kde.score_samples(x)
    for i in range(N):
        kl += log_density[i] - multivariate_normal.logpdf(x[i], mu, sig)        
    return kl/N 



def plugin_estimator(x, lam, r=0):
    kl = 0
    N_is = np.unique(x, return_counts=True)[1]
    x_is = np.unique(x)
    for it in range(len(x_is)):
        temp =  N_is[it]/ len(x)
        kl += temp * (np.log(temp)- poisson.logpmf(x_is[it], lam+r))  
    return kl
