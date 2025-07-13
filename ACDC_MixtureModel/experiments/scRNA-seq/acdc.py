#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from scipy.stats import multivariate_normal
from scipy.special import digamma
from sklearn.mixture import GaussianMixture as GMM
from matplotlib import pyplot as plt
from scipy.stats import uniform
import os
import yaml


class STARE():

    def __init__(self, x):
        self.x = x
        
    def _kl_est(self, sk, muk, vark, scale, bias_correction):
        d = self.x.shape[1]
        kl_k = 0
        N_k = len(sk)
        knn_min = np.unique(sk, return_counts = True, axis = 0)[1][0]+1
        if scale is not None:
            knn = max(int(N_k ** scale), knn_min)
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
    
    def fit(self, maxK, fig_name, scale=1/2, bias_correction=True):
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
                divergence_comp[k]=self._kl_est(self.x[z==k], mus[k], var[k], scale, bias_correction)
                print('divergence for %ith component:'%(k+1), divergence_comp[k])
            loss = sum(np.multiply(pis, divergence_comp))

            kl_sims.append(divergence_comp)
            loss_sims.append(loss)
            z_sims.append(z)
            mus_sims.append(mus)
            sds_sims.append(var) 
            pis_sims.append(pis)
        return z_sims, mus_sims, sds_sims, pis_sims, loss_sims, kl_sims

def plot_loss_against_rho(kls, pis, maxK, lam, fig_name, log=True,legend=True,rho_min = 0, rho_max = 10):
    r = np.linspace(rho_min,rho_max,100)
    fig, axis = plt.subplots(1, 1, figsize=(6,3))
    for pi, kl, K in zip(pis, kls, np.arange(1,maxK)):
        y = np.sum(np.multiply(pi, np.maximum(kl-r[:,np.newaxis],0)),axis = 1) + lam * K
        #print(y)
        axis.plot(r, y, lw=1.8,label = "K=%i"%K)

    axis.set_xlabel(r'$\rho$',fontsize=20)
    axis.set_ylabel('Penalized loss',fontsize=20)
    if log:
        axis.set_yscale('log')
    axis.xaxis.set_tick_params(labelsize=13)
    axis.yaxis.set_tick_params(labelsize=13)
    sns.despine()
    if legend:
        axis.legend(prop={'size': 12})
        fig.savefig(fig_name+'-legend.pdf',bbox_inches='tight')    
    else:
        axis.legend('', frameon=False)
        fig.savefig(fig_name+'.pdf',bbox_inches='tight')

    
sys_id = int(os.getenv('SGE_TASK_ID'))
# config file
with open('/projectnb/mutsigs/menglai/RNAseq/code/configs/config.yml', 'r') as file:
    config = yaml.safe_load(file)

samp_info = config['subsampling']
num_rep = samp_info['num_rep']
#samp_path = samp_info['subsample_path']

if samp_info['is_fixed_ctype']:
    ntype = samp_info['num_type']
else:
    ntype = samp_info['num_type'] * sys_id

if samp_info['is_fixed_cobs']:
    nobs = 250 + samp_info['num_obs']
else:
    nobs = 250 + samp_info['num_obs'] * sys_id

seed = samp_info['seed']

proc_info = config['preprocess']
lab_path = proc_info['label_path']
pca_path = proc_info['pca_proj_path']

sarm_info = config['sarm']
fig_path = sarm_info['fig_path']
K_base = sarm_info['maxK_base']
maxK = K_base + ntype

for i in np.arange(1,num_rep+1):
    file = f"{pca_path}pca_{ntype}type{nobs}rep{i}.csv"
    df = pd.read_csv(file, index_col=0)
    
    x=df.values
    stare = STARE(x)
    z_sims, mus_sims, sds_sims, pis_sims, loss_sims, kl_sims= stare.fit(maxK, scale = 1/2,fig_name=False)
    
    fig_n=f"{fig_path}rho_{ntype}type{nobs}rep{i}"
    N = 1000
    plot_loss_against_rho(kl_sims, pis_sims, maxK=maxK, lam=10/N, rho_min=0, rho_max=20, log=True, legend=True, fig_name=fig_n)

    z_sims_df = pd.DataFrame(z_sims).T
    z_sims_df['cell'] = df.index
    z_sims_df.to_csv(f"{lab_path}SARM_{ntype}type{nobs}rep{i}.csv",index=False)

