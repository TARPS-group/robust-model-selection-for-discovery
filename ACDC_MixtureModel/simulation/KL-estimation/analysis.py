#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains analysis functions.
"""
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.special import digamma
from estimators import knn_estimator,kl_divergence_knn_estimator
from scipy.linalg import det, inv



def generate_correlation(d,sigma):
    # c = np.zeros((d,d))
    # for i in range(d):
    #     for j in range(d):
    #         c[i,j] = np.exp(-(i-j)**2/sigma**2)
    a = np.arange(d)
    c = np.exp(-(a[:,np.newaxis]-a[np.newaxis,:])**2/sigma**2)
    return c   



def multivariate_gaussian_kl(mu1, cov1, mu2, cov2):
    k = len(mu1)  # Dimensionality of the distributions
    
    # Ensure covariance matrices are positive definite
    epsilon = 1e-6
    cov1 += epsilon * np.eye(k)
    cov2 += epsilon * np.eye(k)
    
    # Compute KL divergence
    term1 = 0.5 * (np.trace(np.dot(inv(cov2), cov1)) + 
                   np.dot((mu2 - mu1).T, np.dot(inv(cov2), (mu2 - mu1))) - k)
    term2 = 0.5 * (np.log(det(cov2) / det(cov1)))
    
    return term1 + term2


def compute_KNNest_given_N(mu1, Sigma1, mu2, Sigma2, rates, scales, k_ubs = [10], Ns = 'default', Nmin = 100, Nmax = 20000, l = 10, nrep = 5):
    # k = scale * N^rate
    if Ns != 'default':
        l = len(Ns)
    assert len(scales) == len(rates)
    
    nscale = len(scales)
    nk = len(k_ubs)
    est = np.zeros((l,nrep,nscale))
    est_b = np.zeros((l,nrep,nk))
    est_ub = np.zeros((l,nrep,nk))
    if Ns == 'default':
        Ns = [int(n) for n in np.linspace(start=Nmin, stop=Nmax, num=l)]
        
    for i,N in enumerate(Ns):
        for r in range(nrep):
            x = multivariate_normal.rvs(mu1, Sigma1, size = N)
            for ik, k_ub in enumerate(k_ubs):
                # est_b[i][r][ik] = knn_estimator(x, mu2, Sigma2, k_ub)
                est_b[i][r][ik] = knn_estimator(x, mu2, Sigma2, k_ub)
                est_ub[i][r][ik] = est_b[i][r][ik] - np.log(k_ub) + digamma(k_ub)
            
            for j in range(nscale):
                k = int(scales[j]*N**rates[j])
                est[i][r][j] = knn_estimator(x, mu2, Sigma2, k)
            
    return Ns, est, est_b, est_ub



def generate_df(est, Ns, label_name, df = None, nrep = 5):
    for i in range(len(Ns)):
        df1 = pd.DataFrame({'Number of samples': np.array([Ns[i]]*nrep),
                           'Error': np.array(est[i]),
                           'Method' : np.array([label_name]*nrep)})
        df = pd.concat([df,df1])
    return df


# For discrete case:

# def compute_KDEest_given_N(mu1, Sigma1, mu2, Sigma2, Ns = 'default', bandwidth = 'scott', kernel = 'gaussian', Nmin = 100, Nmax = 100000, l = 10, nrep=5):
#     est = np.zeros((l,nrep))
#     if Ns == 'default':
#         Ns = [int(n) for n in np.linspace(start=Nmin, stop=Nmax, num=l)]
 
#     for i,N in enumerate(Ns):
#         for r in range(nrep):        
#             x = multivariate_normal.rvs(mu1, Sigma1, size = N)
#             est[i][r] = kde_estimator(x, mu2, Sigma2, bandwidth, kernel)
#     return Ns, est


# def compute_true_KL_pois(lam1, lam2):
#     return lam1 * np.log(lam1/lam2) + lam2 - lam1
    
# def compute_pluginKL_given_N(lam1, lam2, r = None, Nmin = 1000, Nmax = 100000, l = 10):
#     est = []
#     Ns = [int(n) for n in np.linspace(start=Nmin, stop=Nmax, num=l)]
#     for N in Ns:
#         if r:
#             x = poisson.rvs(lam1, size = N)+poisson.rvs(r, size = N)
#         else:
#             x = poisson.rvs(lam1, size = N)
#         kl_est = plugin_estimator(x, lam2, r)
#         est.append(kl_est)
#     return Ns, est