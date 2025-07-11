#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from utils import (create_folder_if_not_exist,
                   Timer)
from analysis import (generate_correlation,
                      generate_df,
                      multivariate_gaussian_kl,
                      compute_KNNest_given_N)

'''
This script aims to study KL estimation with varying dimension and sample size.
 a. Continous case: fixed kNN estimator, bias-corrected kNN estimator, 
     adaptive kNN estimator for KL between 
     two multivariate Gaussian distributions
     -- Scenario I: same means, diff covariances
     -- Scenario II: diff means, same covariances
    
 b. Discrete case: plug-in estimator
     for KL between two univariate Poisson distributions
'''   


dir_path = os.getcwd()
create_folder_if_not_exist(dir_path+'/figures')                  
create_folder_if_not_exist(dir_path+'/results')


#######################################
######## weak correlation case ########
#######################################

d = 25
sigma = 0.6
mu1, mu2 = np.array([1]*d), np.array([0]*d)
corr1 = generate_correlation(d, sigma)
corr2 = np.identity(d)
sd = np.diag([1]*d)
Sigma1, Sigma2 = sd@corr1@sd, sd@corr2@sd
true_kl = multivariate_gaussian_kl(mu1, Sigma1, mu2, Sigma2)
print('True KL: %.3f'%true_kl)


########################
#### kNN estimator #####
########################


with Timer('Run multiple KL estimators'): 
    # Ns = [100, 1000, 5000, 10000]
    Ns = [100, 1000, 5000, 10000, 20000, 50000]
    Ns, est_adapt_2, est_b_2, est_ub_2 = compute_KNNest_given_N(mu1, Sigma1, mu2, Sigma2, Ns = Ns, k_ubs = [1,10], rates = [1/2], scales = [1]*1, nrep = 1)
   
np.save('results/weak-corr-adaptive-knnest-d=%i'%(d), est_adapt_2)
np.save('results/weak-corr-fixed-knnest-d=%i'%(d), est_b_2)
np.save('results/weak-corr-fixed-bias-corrected-knnest-d=%i'%(d), est_ub_2)


# Plot knn estimates versus sample size
if True:
    
    est_b_2 = np.load('results/weak-corr-fixed-knnest-d=%i.npy'%(d))
    est_ub_2 = np.load('results/weak-corr-fixed-bias-corrected-knnest-d=%i.npy'%(d))
    est_adapt_2 = np.load('results/weak-corr-adaptive-knnest-d=%i.npy'%(d))

    dat = generate_df(abs(est_b_2[:,:,0]-true_kl).reshape(-1), Ns, label_name = r'$k=1$' , df = None, nrep = 1)
    dat = generate_df(abs(est_ub_2[:,:,0]-true_kl).reshape(-1), Ns, label_name = r'$k=1$ (bias-corrected)', df = dat, nrep = 1)  
    dat = generate_df(abs(est_b_2[:,:,1]-true_kl).reshape(-1), Ns, label_name = r'$k=10$' , df = dat, nrep = 1)
    dat = generate_df(abs(est_ub_2[:,:,1]-true_kl).reshape(-1), Ns, label_name = r'$k=10$ (bias-corrected)', df = dat, nrep = 1)    
    # dat = generate_df(abs(est_adapt_2[:,:,0]-true_kl).reshape(-1), Ns, label_name = r'$k=N^{1/4}$', df = dat, nrep = 1)   
    dat = generate_df(abs(est_adapt_2[:,:,0]-true_kl).reshape(-1), Ns, label_name = r'$k=N^{1/2}$', df = dat, nrep = 1)  
    # dat = generate_df(abs(est_adapt_2[:,:,2]-true_kl).reshape(-1), Ns, label_name = r'$k=N^{3/4}$', df = dat, nrep = 1)    


    sns.set_palette('colorblind')
    # dat = generate_df(abs(est_b_1[:,:,1]-true_kl), Ns, label_name = r'k=30' , df = dat, nrep = 5)
    # dat = generate_df(abs(est_ub_1[:,:,1]-true_kl), Ns, label_name = r'k=30 (bias-corrected)', df = dat, nrep = 5)    
    # dat = generate_df(abs(est_b_1[:,:,2]-true_kl), Ns, label_name = r'k=50' , df = dat, nrep = 5)
    # dat = generate_df(abs(est_ub_1[:,:,2]-true_kl), Ns, label_name = r'k=50 (bias-corrected)', df = dat, nrep = 5)    

    fig,axis=plt.subplots(1,1,figsize=(6, 3)) 
    sns.pointplot(x = 'Number of samples', y = 'Error', hue='Method', data = dat, dodge=0.2, join=True, errorbar = 'ci', errwidth =2, capsize=0.05,ax=axis)
    axis.set_xlabel(xlabel = 'Number of samples', fontsize=20)
    axis.set_ylabel(ylabel = 'Error',fontsize=20)
    axis.xaxis.set_tick_params(labelsize=13)
    axis.yaxis.set_tick_params(labelsize=13)
    axis.get_legend().remove()
    sns.despine()
    axis.axhline(y = 0, ls='--', color = 'gray')
    fig.savefig('figures/knnest-weak-corr-d=%i.pdf'%(d),bbox_inches='tight')  


    fig,axis=plt.subplots(1,1,figsize=(6, 3)) 
    sns.pointplot(x = 'Number of samples', y = 'Error', hue='Method', data = dat, dodge=0.2, join=True, errorbar = 'ci', errwidth =2, capsize=0.05,ax=axis)
    axis.set_xlabel(xlabel = 'Number of samples', fontsize=20)
    axis.set_ylabel(ylabel = 'Error',fontsize=20)
    axis.xaxis.set_tick_params(labelsize=13)
    axis.legend(prop={'size': 12})
    axis.yaxis.set_tick_params(labelsize=13)
    sns.despine()
    axis.axhline(y = 0, ls='--', color = 'gray')
    fig.savefig('figures/knnest-weak-corr-d=%i-legend.pdf'%(d),bbox_inches='tight')  



#################################################
######## Fig 2. in Wang et al (2009) d=4 ########
#################################################

d = 4
mu1, mu2 = np.array([0.1, 0.3, 0.6, 0.9]), np.array([0]*d)
Sigma1, Sigma2 = 0.5*np.identity(d)+0.5, 0.9*np.identity(d)+0.1

true_kl = multivariate_gaussian_kl(mu1, Sigma1, mu2, Sigma2)
print('True KL: %.3f'%true_kl)


########################
#### kNN estimator #####
########################


with Timer('Run multiple KL estimators'): 
    # Ns = [100, 1000, 5000, 10000]
    Ns = [100, 1000, 5000, 10000, 20000, 50000]
    Ns, est_adapt_2, est_b_2, est_ub_2 = compute_KNNest_given_N(mu1, Sigma1, mu2, Sigma2, Ns = Ns, k_ubs = [1,10], rates = [1/2], scales = [1]*1, nrep = 1)
   
np.save('results/fig2-corr-adaptive-knnest-d=%i'%(d), est_adapt_2)
np.save('results/fig2-corr-fixed-knnest-d=%i'%(d), est_b_2)
np.save('results/fig2-corr-fixed-bias-corrected-knnest-d=%i'%(d), est_ub_2)


# Plot knn estimates versus sample size
if True:
    
    est_b_2 = np.load('results/fig2-corr-fixed-knnest-d=%i.npy'%(d))
    est_ub_2 = np.load('results/fig2-corr-fixed-bias-corrected-knnest-d=%i.npy'%(d))
    est_adapt_2 = np.load('results/fig2-corr-adaptive-knnest-d=%i.npy'%(d))

    dat = generate_df(abs(est_b_2[:,:,0]-true_kl).reshape(-1), Ns, label_name = r'$k=1$' , df = None, nrep = 1)
    dat = generate_df(abs(est_ub_2[:,:,0]-true_kl).reshape(-1), Ns, label_name = r'$k=1$ (bias-corrected)', df = dat, nrep = 1)  
    dat = generate_df(abs(est_b_2[:,:,1]-true_kl).reshape(-1), Ns, label_name = r'$k=10$' , df = dat, nrep = 1)
    dat = generate_df(abs(est_ub_2[:,:,1]-true_kl).reshape(-1), Ns, label_name = r'$k=10$ (bias-corrected)', df = dat, nrep = 1)    
    # dat = generate_df(abs(est_adapt_2[:,:,0]-true_kl).reshape(-1), Ns, label_name = r'$k=N^{1/4}$', df = dat, nrep = 1)   
    dat = generate_df(abs(est_adapt_2[:,:,0]-true_kl).reshape(-1), Ns, label_name = r'$k=N^{1/2}$', df = dat, nrep = 1)  
    # dat = generate_df(abs(est_adapt_2[:,:,2]-true_kl).reshape(-1), Ns, label_name = r'$k=N^{3/4}$', df = dat, nrep = 1)    


    sns.set_palette('colorblind')
    # dat = generate_df(abs(est_b_1[:,:,1]-true_kl), Ns, label_name = r'k=30' , df = dat, nrep = 5)
    # dat = generate_df(abs(est_ub_1[:,:,1]-true_kl), Ns, label_name = r'k=30 (bias-corrected)', df = dat, nrep = 5)    
    # dat = generate_df(abs(est_b_1[:,:,2]-true_kl), Ns, label_name = r'k=50' , df = dat, nrep = 5)
    # dat = generate_df(abs(est_ub_1[:,:,2]-true_kl), Ns, label_name = r'k=50 (bias-corrected)', df = dat, nrep = 5)    

    fig,axis=plt.subplots(1,1,figsize=(6, 3)) 
    sns.pointplot(x = 'Number of samples', y = 'Error', hue='Method', data = dat, dodge=0.2, join=True, errorbar = 'ci', errwidth =2, capsize=0.05,ax=axis)
    axis.set_xlabel(xlabel = 'Number of samples', fontsize=20)
    axis.set_ylabel(ylabel = 'Error',fontsize=20)
    axis.xaxis.set_tick_params(labelsize=13)
    axis.yaxis.set_tick_params(labelsize=13)
    axis.get_legend().remove()
    sns.despine()
    axis.axhline(y = 0, ls='--', color = 'gray')
    fig.savefig('figures/knnest-fig2-corr-d=%i.pdf'%(d),bbox_inches='tight')  


    fig,axis=plt.subplots(1,1,figsize=(6, 3)) 
    sns.pointplot(x = 'Number of samples', y = 'Error', hue='Method', data = dat, dodge=0.2, join=True, errorbar = 'ci', errwidth =2, capsize=0.05,ax=axis)
    axis.set_xlabel(xlabel = 'Number of samples', fontsize=20)
    axis.set_ylabel(ylabel = 'Error',fontsize=20)
    axis.xaxis.set_tick_params(labelsize=13)
    axis.legend(prop={'size': 12})
    axis.yaxis.set_tick_params(labelsize=13)
    sns.despine()
    axis.axhline(y = 0, ls='--', color = 'gray')
    fig.savefig('figures/knnest-fig2-corr-d=%i-legend.pdf'%(d),bbox_inches='tight')  




##################################################
######## Fig 3. in Wang et al (2009) d=10 ########
##################################################

d = 10
mu1, mu2 = np.array([0]*d), np.array([0]*d)
Sigma1, Sigma2 = 0.1*np.identity(d)+0.9, 0.9*np.identity(d)+0.1

true_kl = multivariate_gaussian_kl(mu1, Sigma1, mu2, Sigma2)
print('True KL: %.3f'%true_kl)


########################
#### kNN estimator #####
########################


with Timer('Run multiple KL estimators'): 
    # Ns = [100, 1000, 5000, 10000]
    Ns = [100, 1000, 5000, 10000, 20000, 50000]
    Ns, est_adapt_2, est_b_2, est_ub_2 = compute_KNNest_given_N(mu1, Sigma1, mu2, Sigma2, Ns = Ns, k_ubs = [1,10], rates = [1/2], scales = [1]*1, nrep = 1)
 
np.save('results/fig3-corr-adaptive-knnest-d=%i'%(d), est_adapt_2)
np.save('results/fig3-corr-fixed-knnest-d=%i'%(d), est_b_2)
np.save('results/fig3-corr-fixed-bias-corrected-knnest-d=%i'%(d), est_ub_2)


# Plot knn estimates versus sample size
if True:
    
    est_b_2 = np.load('results/fig3-corr-fixed-knnest-d=%i.npy'%(d))
    est_ub_2 = np.load('results/fig3-corr-fixed-bias-corrected-knnest-d=%i.npy'%(d))
    est_adapt_2 = np.load('results/fig3-corr-adaptive-knnest-d=%i.npy'%(d))

    dat = generate_df(abs(est_b_2[:,:,0]-true_kl).reshape(-1), Ns, label_name = r'$k=1$' , df = None, nrep = 1)
    dat = generate_df(abs(est_ub_2[:,:,0]-true_kl).reshape(-1), Ns, label_name = r'$k=1$ (bias-corrected)', df = dat, nrep = 1)  
    dat = generate_df(abs(est_b_2[:,:,1]-true_kl).reshape(-1), Ns, label_name = r'$k=10$' , df = dat, nrep = 1)
    dat = generate_df(abs(est_ub_2[:,:,1]-true_kl).reshape(-1), Ns, label_name = r'$k=10$ (bias-corrected)', df = dat, nrep = 1)    
    # dat = generate_df(abs(est_adapt_2[:,:,0]-true_kl).reshape(-1), Ns, label_name = r'$k=N^{1/4}$', df = dat, nrep = 1)   
    dat = generate_df(abs(est_adapt_2[:,:,0]-true_kl).reshape(-1), Ns, label_name = r'$k=N^{1/2}$', df = dat, nrep = 1)  
    # dat = generate_df(abs(est_adapt_2[:,:,2]-true_kl).reshape(-1), Ns, label_name = r'$k=N^{3/4}$', df = dat, nrep = 1)    


    sns.set_palette('colorblind')
    # dat = generate_df(abs(est_b_1[:,:,1]-true_kl), Ns, label_name = r'k=30' , df = dat, nrep = 5)
    # dat = generate_df(abs(est_ub_1[:,:,1]-true_kl), Ns, label_name = r'k=30 (bias-corrected)', df = dat, nrep = 5)    
    # dat = generate_df(abs(est_b_1[:,:,2]-true_kl), Ns, label_name = r'k=50' , df = dat, nrep = 5)
    # dat = generate_df(abs(est_ub_1[:,:,2]-true_kl), Ns, label_name = r'k=50 (bias-corrected)', df = dat, nrep = 5)    

    fig,axis=plt.subplots(1,1,figsize=(6, 3)) 
    sns.pointplot(x = 'Number of samples', y = 'Error', hue='Method', data = dat, dodge=0.2, join=True, errorbar = 'ci', errwidth =2, capsize=0.05,ax=axis)
    axis.set_xlabel(xlabel = 'Number of samples', fontsize=20)
    axis.set_ylabel(ylabel = 'Error',fontsize=20)
    axis.xaxis.set_tick_params(labelsize=13)
    axis.yaxis.set_tick_params(labelsize=13)
    axis.get_legend().remove()
    sns.despine()
    axis.axhline(y = 0, ls='--', color = 'gray')
    fig.savefig('figures/knnest-fig3-corr-d=%i.pdf'%(d),bbox_inches='tight')  


    fig,axis=plt.subplots(1,1,figsize=(6, 3)) 
    sns.pointplot(x = 'Number of samples', y = 'Error', hue='Method', data = dat, dodge=0.2, join=True, errorbar = 'ci', errwidth =2, capsize=0.05,ax=axis)
    axis.set_xlabel(xlabel = 'Number of samples', fontsize=20)
    axis.set_ylabel(ylabel = 'Error',fontsize=20)
    axis.xaxis.set_tick_params(labelsize=13)
    axis.legend(prop={'size': 12})
    axis.yaxis.set_tick_params(labelsize=13)
    sns.despine()
    axis.axhline(y = 0, ls='--', color = 'gray')
    fig.savefig('figures/knnest-fig3-corr-d=%i-legend.pdf'%(d),bbox_inches='tight')  

