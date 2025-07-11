#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import multivariate_normal
from acdc import ACDC
from utils import Timer, check_if_data_exists, create_folder_if_not_exist
from visualization import (plot_fmeasure_against_K, plot_loss_against_rho)


dir_path = os.getcwd()
create_folder_if_not_exist(dir_path+'/datasets')
create_folder_if_not_exist(dir_path+'/results')
create_folder_if_not_exist(dir_path+'/figures')


def generate_correlation(d,sigma):
    # c = np.zeros((d,d))
    # for i in range(d):
    #     for j in range(d):
    #         c[i,j] = np.exp(-(i-j)**2/sigma**2)
    a = np.arange(d)
    c = np.exp(-(a[:,np.newaxis]-a[np.newaxis,:])**2/sigma**2)
    return c   

def cell_types_model_probit(M, N, mus, Sigmas, pmix = None):
    K_t = len(mus)
    if pmix is None:
        pmix = np.random.uniform(size = K_t)
        pmix /= sum(pmix)
    J = np.zeros((M,N))
    z_true = np.zeros(M)
    for i in range(M):
        k = np.random.choice(a = K_t, p = pmix)
        z_true[i] = k
        J[i] = multivariate_normal.rvs(mus[k], Sigmas[k], 1) > 0

    return np.array(J, dtype = np.int32), np.array(z_true, dtype = np.int32), pmix



K_t = 4
M, N = 10000, 10
p0 = [0.3, 0.3, 0.2, 0.2]
s0 = [0.2, 0.2, 0.2, 0.2] 
mu0 = multivariate_normal.rvs([0]*N, np.identity(N), size = K_t)
Sigma0 = np.zeros((K_t,N,N))
for k in range(K_t):
    corr = generate_correlation(N, s0[k])
    print('corr %i:'%k, corr)
    sd = np.diag([1]*N)
    Sigma0[k] = sd@corr@sd
mu0


data_path = "datasets/simulated-scRNA-Kt=%i-M=%i-N=%i.json"%(K_t,M,N)

if check_if_data_exists(data_path):
    print("Reading data...")
    with open(data_path, 'r', encoding='utf8') as f:
        dat = json.load(f)
    x = np.array(dat['x_t'])
else:
    print("Creating data...")
    x, z_t, p_t = cell_types_model_probit(M, N, mu0, Sigma0, p0)
    cur_data = {'x_t':x.tolist(), 'z_t': z_t.tolist(),'p_t':p_t}
    with open(data_path, 'w', encoding='utf8') as f:
        json.dump(cur_data, f)

with Timer('runing ACDC'):
    maxK = 7
    caliAcdc = ACDC(x)
    z_sims,mus_sims,pis_sims,kl_sims,loss_sims,split_kth_sims= caliAcdc.fit( maxK = maxK, iter_update = 30, eps = 1e-2)
    



plot_fmeasure_against_K(z_sims, z_t, K_t, fig_name = 'figures/mosquito-fmeasure-plot-Kt=%i-M=%i-N=%i'%(K_t,M,N))
plot_loss_against_rho(kl_sims, pis_sims, maxK = maxK, lam = 1/M, rho_min = 0, rho_max = 1, log = True, fig_name = 'figures/mosquito-loss-plot-Kt=%i-M=%i-N=%i'%(K_t,M,N))
