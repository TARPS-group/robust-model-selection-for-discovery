#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:24:27 2023

@author: jwli
"""
import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from acdc import ACDC
from data import DSSkewMixture
from visualization import  plot_loss_against_rho
from scipy.stats import multivariate_normal, norm
from sklearn.mixture import GaussianMixture as GMM
from utils import Timer, check_if_data_exists, create_folder_if_not_exist


dir_path = os.getcwd()
create_folder_if_not_exist(dir_path+'/datasets')
create_folder_if_not_exist(dir_path+'/results')
create_folder_if_not_exist(dir_path+'/figures')

# True distribution paramters
K = 2
sds = [1, 1]
xmin, xmax = -10, 10

close = False
relative_size = "big-small"
relative_mis = "big-small"    
N = 10000


## Generate data
if True:
    pmix = [0.95, 0.05] if relative_size == "big-small" else [0.5, 0.5]
    if relative_mis == "small-big":
        a = [-1, -10]
    elif relative_mis == "big-small":
        a = [-10, -1]
    elif relative_mis == "bbig-small":
        a = [-100,-1]
    elif relative_mis == "equal":
        a = [-10, -10]
    elif relative_mis == "none":
        a = [0, 0]
    means = [-3, 0] if close else [-3, 3]
    data = DSSkewMixture(K, means, sds, a, pmix)
    data_path = "datasets/n=%i-close-%s-rsize-%s-rmis-%s.json"%(N,close,relative_size,relative_mis)

    if check_if_data_exists(data_path):
        print("Reading data...")
        with open(data_path, 'r', encoding='utf8') as f:
            dat = json.load(f)
        x = np.array(dat['x_t'])
    else:
        print("Creating data...")
        x = data.sample(N = N)
        z = data.true_cluster()
        cur_data = {'x_t':x.tolist(), 'z_t': z.tolist()}
        with open(data_path, 'w', encoding='utf8') as f:
            json.dump(cur_data, f)


##############################
#####  ACDC Inference  ######
##############################  
maxK = 8
  
## Running ACDC
if True:
    x = np.array(x).reshape(-1,1)    

    with Timer('ACDC'):
        N_eff = int(np.sqrt(N))+1
        acdc = ACDC(x)
        z_sims, mus_sims, sds_sims, pis_sims, loss_sims, kl_sims= acdc.fit(maxK, scale = 1/2, fig_name = 'figures/ACDC-n=%i-close-%s-rsize-%s-rmis-%s'%(N,close,relative_size,relative_mis))



cur_res = {'z_sims': [l.tolist() for l in z_sims], 'mus_sims': [l.tolist() for l in mus_sims], 'sds_sims':[l.tolist() for l in sds_sims],'pis_sims':[l.tolist() for l in pis_sims],
           'kl_sims':[l.tolist() for l in kl_sims], 'loss_sims':loss_sims}
with open("results/acdc-n=%.f-close-%s-rsize-%s-rmis-%s.json"%(N,close,relative_size,relative_mis), 'w', encoding='utf8') as f:
    json.dump(cur_res, f)


###########################
###  Plot ACDC results ###
###########################

with open("results/acdc-n=%.f-close-%s-rsize-%s-rmis-%s.json"%(N,close,relative_size,relative_mis), 'r', encoding='utf8') as f:    
     res = json.load(f)

kl_sims = res['kl_sims']
pis_sims = res['pis_sims']

    
plot_loss_against_rho(kl_sims, pis_sims, maxK = 5, lam = 0.01, rho_min = 0, rho_max = 2, log=True, legend=False, fig_name = 'figures/skewnorm-acdc-loss-n=%i-close-%s-rsize-%s-rmis-%s'%(N,close,relative_size,relative_mis))
plot_loss_against_rho(kl_sims, pis_sims, maxK = 5, lam = 0.01, rho_min = 0, rho_max = 2, log=True, legend=True, fig_name = 'figures/skewnorm-acdc-loss-n=%i-close-%s-rsize-%s-rmis-%s'%(N,close,relative_size,relative_mis))



if True:
    xmin, xmax = -7.5, 7.5
    grid = np.arange(xmin, xmax, 0.01)
    truth = np.zeros(len(grid))  
    for k in range(data.ncomp()):
        truth += data.pdf(grid, k)
               
    mus_sims = res['mus_sims']
    sds_sims = res['sds_sims']
    
    s_pmix = pis_sims[1]
    s_mus = mus_sims[1]
    s_covs = sds_sims[1]
    s_K = 2
               
    fig, axis = plt.subplots(1, 1, figsize=(6,3))
    # for k in range(K):
    #     comps = data.pdf(grid, k)
    #     axis.plot(grid, comps, 'black', linestyle = 'dotted',lw=2)  
        
    s_density = np.zeros(len(grid))  
    for k in range(s_K):
        s_density += s_pmix[k] * multivariate_normal.pdf(grid.reshape(-1,1), s_mus[k], s_covs[k])
    axis.plot(grid, truth, 'black', linestyle = 'dashed', lw=3, label ="true distribution" )
    axis.plot(grid, s_density, 'blue', linestyle = '-', lw=2.5, label ="ACDC" )
    
    for k in range(s_K):
        s_comps = s_pmix[k] * multivariate_normal.pdf(grid.reshape(-1,1), s_mus[k], s_covs[k])
        axis.plot(grid, s_comps, linestyle = '-',lw=1)  
    
    #axis.legend('', frameon=False)
    #axis.legend(loc = 'upper right', prop={'size': 16})
    axis.set_title("Our method / small-large (K=%i)"%K, size = 20)
    axis.set_xlabel("x", size = 20)
    axis.set_ylabel("Density", size = 20)
    axis.set_ylim(0,0.85)
    axis.set_xlim(-7,5.5)
    axis.xaxis.set_tick_params(labelsize=16)
    axis.yaxis.set_tick_params(labelsize=16)
    sns.despine()
    fig.tight_layout(h_pad=1)
    fig.savefig('figures/skewnorm-acdc-pdfs-n=%i-close-%s-rsize-%s-rmis-%s.pdf'%(N,close,relative_size,relative_mis))


####################################
#####  Coarsening  Inference  ######
####################################  
# Plot calibration plots

with open("results/calibration-coarsening-results-n=%.f-close-%s-rsize-%s-rmis-%s.json"%(N,close,relative_size,relative_mis), 'r', encoding='utf8') as f:    
     res = json.load(f)

Ek = res['Ek']
El = res['El']
alphas = res['alphas']
fig, axis = plt.subplots(1, 1, figsize=(6,3))
axis.plot(Ek, El, 'ko', markersize=3)
axis.plot(Ek, El, 'k-')

opt_ind = 9
x_position, y_position = Ek[opt_ind], El[opt_ind] 
alpha_opt = alphas[opt_ind]

fig, axis = plt.subplots(1, 1, figsize=(6,3))
axis.plot(Ek, El, 'ko', markersize=3)
axis.plot(Ek, El, 'k-')
axis.annotate('X', xy=(x_position, y_position), color='red', fontsize=14, ha='center', va='center', weight='bold')
axis.annotate(r'$\alpha = %i$'%round(alphas[opt_ind]), xy=(x_position+0.5, y_position-150), fontsize=16, ha='center', va='center')
axis.set_ylabel(r"$\hat{\mathrm{E}}_{\alpha}\mathrm{(loglik | data)}$",fontsize=20)
axis.set_xlabel(r"$\hat{\mathrm{E}}_{\alpha}(k_{2\%} | \mathrm{data})$",fontsize=20)
axis.set_title("Coarsened calibration / different", size = 20)
sns.despine()
fig.savefig('figures/skewnorm-coarsen-calibration-n=%i-close-%s-rsize-%s-rmis-%s.pdf'%(N,close,relative_size,relative_mis),bbox_inches='tight')


# Plot coarsened results
if True:
    with open("results/coarsening-n=%.f-close-%s-rsize-%s-rmis-%s.json"%(N,close,relative_size,relative_mis), 'r', encoding='utf8') as f:    
         res = json.load(f)
    # plot_separate_pdfs_coarsening(data, res, 'figures/test-n=%i.png'%N, xmin=-10, xmax= 10)
    
    
    Ncomps = res['Ncomps']
    active_comps = list(np.nonzero(Ncomps)[0])
    
    c_pmix = np.array(res['pis'])[active_comps]
    c_mus = np.array(res['mus'])[active_comps]
    c_sds = np.array(res['sds'])[active_comps]
    c_K = res['K']
    
    # Plot true components and density
    truth = np.zeros(len(grid))  
    for k in range(data.ncomp()):
        truth += data.pdf(grid, k)
        
               
    fig, axis = plt.subplots(1, 1, figsize=(6,3))
        
    c_density = np.zeros(len(grid))  
    for k in range(c_K):
        c_comps = c_pmix[k] * norm.pdf(grid, c_mus[k], c_sds[k])
        c_density += c_comps
    
    axis.plot(grid, truth, 'black', linestyle = 'dashed', lw=3, label ="true distribution" )
    axis.plot(grid, c_density, 'blue', linestyle = '-', lw=2.5, label ="coarsened" )
    for k in range(c_K):
        c_comps = c_pmix[k] * norm.pdf(grid, c_mus[k], c_sds[k])
        axis.plot(grid, c_comps, linestyle = '-',lw=1.5)  
    
    #axis.legend(loc = 'upper right', prop={'size': 16})
    axis.legend('', frameon=False)
    axis.set_xlabel("x", size = 20)
    axis.set_title("Coarsened / different (K=%i)"%(c_K), size = 20)
    axis.set_ylabel("Density", size = 20)
    axis.set_ylim(0,0.85) #large-small/ large-large
    axis.set_xlim(-7,5.5)
    # axis.set_ylim(0,0.55) # small-large
    # axis.set_xlim(-7,5.5)
    # axis.set_ylim(0,0.001) # small-large test
    # axis.set_xlim(-1,4)
    axis.xaxis.set_tick_params(labelsize=16)
    axis.yaxis.set_tick_params(labelsize=16)
    sns.despine()
    fig.tight_layout(h_pad=1)
    fig.savefig('figures/skewnorm-coarsen-pdfs-n=%i-close-%s-rsize-%s-rmis-%s.pdf'%(N,close,relative_size,relative_mis))



##################################
###  Plot illustrative figures ###
##################################

# 1.loss vs. K
maxK = 8
kl_flat=[element for sublist in kl_sims for element in sublist]
# l = sorted(kl_flat)[:10]
l = [0.001, 0.01]
line_styles = ['--', ':', '-.']

fig, axis = plt.subplots(1, 1, figsize=(6,3))
for i in range(len(l)):
    ll = l[i]
    lam=ll/N
    r = 0.5
    y = []
    for pi, kl, K in zip(pis_sims, kl_sims, np.arange(1,maxK)):
        y.append(np.sum(np.multiply(pi, np.maximum(np.array(kl)-r,0)) + ll * K))
    # axis.plot(np.arange(1,maxK), y, lw=1.8,label=r"$\lambda$=%.3f"%ll)
    
    axis.plot(np.arange(1,maxK), y, lw=1.8, c='black',ls=line_styles[i],label=r"$\lambda$=%.3f"%ll)


axis.legend(prop={'size': 12})
axis.vlines(x = 2, label = 'Truth', ymin=-0.05, ymax=1.1*max(y),lw=1.2, color='red', ls = '-')
axis.set_xlabel("K", size = 20)
axis.set_ylabel('Penalized loss', size = 20)
#axis.set_ylabel(r"$\mathcal{R}^{\rho,\lambda}(\phi^{(K)};x_{1:N})$", size = 18)
axis.xaxis.set_tick_params(labelsize=13)
axis.yaxis.set_tick_params(labelsize=13)
axis.set_yscale('log')
sns.despine()
fig.tight_layout(h_pad=1)
fig.savefig('figures/pen-acdc-loss-legend.pdf')
# axis.legend('', frameon=False)
# fig.savefig('figures/pen-acdc-loss.pdf')



# 2. loss vs. rho
m = ['s','.','^']
line_styles = [':', '-.','--', (0, (3, 1, 1, 1, 1, 1))]
fig, axis = plt.subplots(1, 1, figsize=(6,3))

for K in [1,3]:
    r = np.linspace(0,0.5,100)
    r_critical = np.append(np.array(kl_sims[K-1]), (r[0],r[-1]))
    y = np.sum(np.multiply(pis_sims[K-1], np.maximum(kl_sims[K-1]-r[:,np.newaxis],0)),axis = 1)
    y_critical = np.sum(np.multiply(pis_sims[K-1], np.maximum(kl_sims[K-1]-r_critical[:,np.newaxis],0)),axis = 1)
    axis.plot(r, y, c='black',ls=line_styles[K-1],lw=2,label = "K=%i"%K)
    axis.scatter(r_critical, y_critical, color='black',marker=m[K-1])
axis.set_xlabel(r'$\rho$',fontsize=20)
axis.set_ylabel('Penalized loss',fontsize=20)
    #axis.set_ylabel(r"$\mathcal{R}^{\rho}(\phi^{(K)};x_{1:N})$", size = 18)
axis.xaxis.set_tick_params(labelsize=14)
axis.yaxis.set_tick_params(labelsize=14)
sns.despine()
fig.tight_layout(h_pad=1)
    #axis.legend('', frameon=False)
fig.savefig('figures/pen-acdc-loss-linear-in-rho.pdf')  




##############################
#####    EM  Inference  ######
##############################  


## Running EM
n = 5000
x = data.sample(N = n)
z = data.true_cluster()
if True:
    n_components = np.arange(1, 21)
    models = [GMM(n, covariance_type='full', random_state=0).fit(x.reshape(-1,1)) for n in n_components]
    bic = [m.bic(x.reshape(-1,1)) for m in models]
    bic_n = n_components[np.where(bic == np.min(bic))]
    aic = [m.aic(x.reshape(-1,1)) for m in models]
    aic_n = n_components[np.where(aic == np.min(aic))]
    gmm = GMM(n_components=bic_n[0]).fit(x.reshape(-1,1))
    z_em = gmm.predict(x.reshape(-1,1))
    em_pmix = gmm.weights_
    em_mus = gmm.means_
    em_covs = gmm.covariances_
    em_K = bic_n[0]



# Plot true components and density for EM
if True:
    xmin, xmax = -7.5, 7.5
    grid = np.arange(xmin, xmax, 0.01)
    truth = np.zeros(len(grid))  
    for k in range(data.ncomp()):
        truth += data.pdf(grid, k)
        
               
    fig, axis = plt.subplots(1, 1, figsize=(6,3))
    # for k in range(K):
    #     comps = data.pdf(grid, k)
    #     axis.plot(grid, comps, 'black', linestyle = 'dotted',lw=2)  
        
    em_density = np.zeros(len(grid))  
    for k in range(em_K):
        em_density += em_pmix[k] * multivariate_normal.pdf(grid.reshape(-1,1), em_mus[k], em_covs[k])
    axis.plot(grid, truth, 'black', linestyle = 'dashed', lw=3, label ="true distribution" )
    axis.plot(grid, em_density, 'blue', linestyle = '-', lw=2.5, label ="EM (BIC)" )
    
    for k in range(em_K):
        em_comps = em_pmix[k] * multivariate_normal.pdf(grid.reshape(-1,1), em_mus[k], em_covs[k])
        axis.plot(grid, em_comps, linestyle = '-',lw=1)  
    
    
    #axis.legend(loc = 'upper right', prop={'size': 16})
    axis.legend('', frameon=False)
    # axis.set_title("BIC: N=%i"%(n), size = 24, fontweight='bold')
    axis.set_title("BIC (EM) / same", size = 20)
    axis.set_xlabel("x", size = 20)
    axis.set_ylabel("Density", size = 20)
    axis.set_ylim(0,0.85)
    axis.xaxis.set_tick_params(labelsize=16)
    axis.yaxis.set_tick_params(labelsize=16)
    sns.despine()
    fig.tight_layout(h_pad=1)
    fig.savefig('figures/em-pdfs-n=%i-close-%s-rsize-%s-rmis-%s.pdf'%(n,close,relative_size,relative_mis))



# WAIC criterion
if True:
    n_components = np.arange(1, 21)
    waic_like = np.zeros(len(n_components))
    for nn,ncomp in enumerate(n_components):
        gmm = GMM(ncomp, covariance_type='full', random_state=0).fit(x.reshape(-1,1)) 
    # Sampling from completed-data distribution
        n_samples = 3000  # Number of samples to draw
        samples = gmm.sample(n_samples)
        sample_data = samples[0]
        log_likelihoods = []
        for i in range(n_samples):
            ll = np.sum(np.log(np.sum([gmm.weights_[j] * multivariate_normal.pdf(sample_data[i], gmm.means_[j], gmm.covariances_[j]) for j in range(ncomp)], axis=0)))
            log_likelihoods.append(ll)
    
        log_likelihoods = np.array(log_likelihoods)
        waic_like[nn] = -2 * (log_likelihoods.mean() - log_likelihoods.var())
    
    waic_n = n_components[np.where(waic_like == np.min(waic_like))]
    
    gmm_waic = GMM(n_components=waic_n[0]).fit(x.reshape(-1,1))
    z_waic = gmm_waic.predict(x.reshape(-1,1))
    pmix_waic = gmm_waic.weights_
    mus_waic = gmm_waic.means_
    covs_waic = gmm_waic.covariances_
    K_waic = waic_n[0]


# Plot true components and density for EM-WAIC
if True:
    xmin, xmax = -7.5, 7.5
    grid = np.arange(xmin, xmax, 0.01)
    truth = np.zeros(len(grid))  
    for k in range(data.ncomp()):
        truth += data.pdf(grid, k)
        
               
    fig, axis = plt.subplots(1, 1, figsize=(6,3))
    # for k in range(K):
    #     comps = data.pdf(grid, k)
    #     axis.plot(grid, comps, 'black', linestyle = 'dotted',lw=2)  
        
    em_density_waic = np.zeros(len(grid))  
    for k in range(K_waic):
        em_density_waic += pmix_waic[k] * multivariate_normal.pdf(grid.reshape(-1,1), mus_waic[k], covs_waic[k])
    axis.plot(grid, truth, 'black', linestyle = 'dashed', lw=3, label ="true distribution" )
    axis.plot(grid, em_density_waic, 'blue', linestyle = '-', lw=2.5, label ="EM (WAIC)" )
    
    for k in range(K_waic):
        em_comps = pmix_waic[k] * multivariate_normal.pdf(grid.reshape(-1,1), mus_waic[k], covs_waic[k])
        axis.plot(grid, em_comps, linestyle = '-',lw=1)  
    
    
    #axis.legend(loc = 'upper right', prop={'size': 16})
    axis.legend('', frameon=False)
    axis.set_title("WAIC: N=%i"%(n), size = 24, fontweight='bold')
    axis.set_xlabel("x", size = 20)
    axis.set_ylabel("Density", size = 20)
    axis.set_ylim(0,0.85)
    axis.xaxis.set_tick_params(labelsize=16)
    axis.yaxis.set_tick_params(labelsize=16)
    sns.despine()
    fig.tight_layout(h_pad=1)
    fig.savefig('figures/em-waic-pdfs-n=%i-close-%s-rsize-%s-rmis-%s.pdf'%(n,close,relative_size,relative_mis))



# Plot AIC, BIC, WAIC

fig, axis = plt.subplots(1, 3, figsize=(8,3))
axis[0].plot(n_components, aic, label ="K_AIC = %i"%aic_n )
axis[1].plot(n_components, bic, label ="K_BIC= %i"%aic_n)
axis[2].plot(n_components, waic_like, label ="K_WAIC= %i"%waic_n)
axis[0].set_xlabel("K", size = 13)
axis[0].set_ylabel("AIC", size = 13)
axis[0].legend()
axis[1].set_xlabel("K", size = 13)
axis[1].set_ylabel("BIC", size = 13)
axis[1].legend()
axis[2].set_xlabel("K", size = 13)
axis[2].set_ylabel("WAIC", size = 13)
axis[2].legend()
fig.tight_layout(h_pad=1)
sns.despine()







