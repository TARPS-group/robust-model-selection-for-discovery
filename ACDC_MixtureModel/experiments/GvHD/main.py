#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from acdc import ACDC
from utils import Timer, create_folder_if_not_exist
from visualization import (plot_true_cluster, 
                           plot_loss_against_rho, 
                           plot_fmeasure_against_K,
                           plot_loss_against_rho_all,
                           plot_em_cluster)
from analysis import F_measure

dir_path = os.getcwd()
create_folder_if_not_exist(dir_path+'/figures')

#########################
####  I. load data  #####
#########################

dat_id = 8
s = pd.read_csv("data/sample%.f.csv"%(dat_id))
label = pd.read_csv("data/labels%.f.csv"%(dat_id), header = None)
raw_X = s.iloc[:,2:]
X, z_true = np.array(raw_X), np.array(label.iloc[:, 0])
K_true = len(np.unique(z_true))
print('True K=%i'%K_true)
w_true = np.unique(z_true, return_counts = True)[1]/len(X)
mu_true = np.array([np.mean(X[z_true == k, :], axis = 0) for k in np.unique(z_true)])
cov_true = np.array([np.cov(X[z_true == k, :].T) for k in np.unique(z_true)])


# explortary analysis 

print('dataset size', X.shape)
plot_true_cluster(s, z_true,  figname = "figures/cluster-plot-true-dataset-%.f.pdf"%(dat_id))


fig = plt.figure()
ax = plt.axes(projection='3d')
x, y, z, h = raw_X.loc[:,'FL1.H'], raw_X.loc[:,'FL2.H'], raw_X.loc[:,'FL3.H'], raw_X.loc[:,'FL4.H']
# cmap = LinearSegmentedColormap.from_list("", ["black","#fde725","#5ec962","#21918c","#3b528b", "#440154"])
cmap = LinearSegmentedColormap.from_list("", ["black","#fde725","#35b779","#31688e","#440154"])
ax.scatter(x,y,z,marker='.', s = 10, c=z_true, cmap=cmap, alpha = 0.3)
ax.set_xlabel('FL1.H')  
ax.set_ylabel('FL2.H')  
ax.set_zlabel('FL3.H')  
fig.savefig('figures/data%i-scatter.pdf'%dat_id)

#########################
####   II. ACDC    #####
#########################
from sklearn.mixture import GaussianMixture as GMM
n = len(X)
if True:
    n_components = np.arange(1, 21)
    models = [GMM(n, covariance_type='full', random_state=0).fit(X) for n in n_components]
    bic = [m.bic(X) for m in models]
    bic_n = n_components[np.where(bic == np.min(bic))]
    gmm = GMM(n_components=bic_n[0]).fit(X)
    z_em = gmm.predict(X)
    em_pmix = gmm.weights_
    em_mus = gmm.means_
    em_covs = gmm.covariances_
    em_K = bic_n[0]
plot_em_cluster(s, z_em,  figname = "figures/cluster-plot-em-dataset-%.f.pdf"%(dat_id))
  
    
#########################
####   II. ACDC    #####
#########################

N = len(z_true)
maxK = 9
N_eff = int(np.sqrt(N))+1
acdc = ACDC(X, noisy = True)
with Timer('ACDC'):
    z_sims,mus_sims,sds_sims,pis_sims,kl_sims,loss_sims,split_kth_sims= acdc.fit(N_eff, scale = 1/2, fig_name = 'figures/GvHD%i-aveEM'%dat_id, maxK = maxK, iter_update = 10, eps = 1e-2, averaged_em = True)

create_folder_if_not_exist(dir_path+'/calibration-results')
np.save('calibration-results/data%i-ACDC-z_sims'%dat_id, z_sims)
np.save('calibration-results/data%i-ACDC-pis_sims'%dat_id, pis_sims)
np.save('calibration-results/data%i-ACDC-kl_sims'%dat_id, kl_sims)
np.save('calibration-results/data%i-ACDC-z_true'%dat_id, z_true)



############################
#### III. analysis #########
############################
def compute_fm_for_rho(pis, kls, zs, z_true, lam, nr=100, rho_min=0, rho_max=5):
    N = len(z_true)
    lam/=N
    r = np.linspace(rho_min,rho_max,nr)
    pen_loss = []
    for pi, kl, K in zip(pis, kls, np.arange(1,maxK)):
        pen_loss.append(np.sum(np.multiply(pi, np.maximum(kl-r[:,np.newaxis],0)),axis = 1) + lam * K)
    cls = np.argmin(pen_loss, axis = 0)
    
    f = np.zeros(nr)

    for i,k in enumerate(cls):
        f[i] = F_measure(z_true, zs[k])
    return r,f

nr = 100
l = 10
fs = np.zeros((6,nr))
for d in range(1,7):
    pis = np.load('calibration-results/data%i-ACDC-pis_sims.npy'%d, allow_pickle=True)
    kls = np.load('calibration-results/data%i-ACDC-kl_sims.npy'%d, allow_pickle=True)
    zt = np.load('calibration-results/data%i-ACDC-z_true.npy'%d, allow_pickle=True)
    zs = np.load('calibration-results/data%i-ACDC-z_sims.npy'%d, allow_pickle=True)
    rs,fs[d-1] = compute_fm_for_rho(pis, kls, zs, zt, lam=l, nr=nr, rho_min=0, rho_max=5)


sns.set_palette('colorblind')
favg = np.mean(fs,axis=0)    
opt_ind = np.argmax(favg)
rho_opt = rs[opt_ind]
fig, axis = plt.subplots(1, 1, figsize=(8,6))
for d in range(6):
    axis.plot(rs, fs[d], label = '%i'%(d+1))
axis.plot(rs, favg, lw =2, linestyle='dashed', color = 'black', label = 'average')
axis.axvline(x=rho_opt, lw =2, color='black')
axis.annotate(r'$\rho \approx$%.2f'%rho_opt, xy=(rho_opt+0.7, 0.995), color='black', fontsize=18, ha='center', va='center')
axis.set_xlabel(r'$\rho$',fontsize=20)
axis.set_ylabel('F-measure',fontsize=20)
axis.xaxis.set_tick_params(labelsize=13)
axis.yaxis.set_tick_params(labelsize=13)
# axis.legend(fontsize = 14)
sns.despine()
fig.savefig('figures/GvHD-train.pdf',bbox_inches='tight')    


############################
#### IV. visualizations ###
############################
pis_sims = np.load('calibration-results/data%i-ACDC-pis_sims.npy'%dat_id, allow_pickle=True)
kl_sims = np.load('calibration-results/data%i-ACDC-kl_sims.npy'%dat_id, allow_pickle=True)
z_sims = np.load('calibration-results/data%i-ACDC-z_sims.npy'%dat_id, allow_pickle=True)

rho_opt = 1.161616
maxK = 9

plot_loss_against_rho_all(kl_sims, pis_sims, K_true, rho_opt, maxK = maxK, lam = 10, rho_min = 0, rho_max = 5, log=True, fig_name = 'figures/GvHD%i-all-loss-plot'%dat_id)
plot_loss_against_rho_all(kl_sims, pis_sims, K_true, rho_opt, maxK = maxK, lam = 10, rho_min = 0, rho_max = 5, log=True, legend = False, fig_name = 'figures/GvHD%i-all-loss-plot'%dat_id)
K_select = 2

plot_loss_against_rho(kl_sims, pis_sims, K_true, K_select, rho_opt, maxK = maxK, lam = 10, rho_min = 0, rho_max = 3, log=True, fig_name = 'figures/GvHD%i-loss-plot'%dat_id)
plot_loss_against_rho(kl_sims, pis_sims, K_true, K_select, rho_opt, maxK = maxK, lam = 10, rho_min = 0, rho_max = 3, log=True, legend = False, fig_name = 'figures/GvHD%i-loss-plot'%dat_id)

K_select = 3
plot_fmeasure_against_K(z_sims, z_true, K_true, K_select, fig_name = 'figures/GvHD%i-fmeasure-plot'%dat_id, legend=True)
plot_fmeasure_against_K(z_sims, z_true, K_true, K_select, fig_name = 'figures/GvHD%i-fmeasure-plot'%dat_id, legend=False)

