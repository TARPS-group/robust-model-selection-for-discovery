# Simulation to calibrate and clustering using coarsened mixture
# Input: simulated data from "/data", best calibrated alpha from "/results/calibration-results-%"
# Output: coarsening results to "/results"
# ________________________________________________________________________________________________________
# Before running include("run-comparison.jl"), change current directory to current path
cd("/Users/jwli/Documents/GitHub/robust-mixture-model-selection/code/simulation/gaussian-mixture-model")

## loading packages and functions
using Distributions
using PyPlot
using JSON

include("coarsening/helper.jl") # Code for helper functions
include("coarsening/core.jl") # Code for mixture model MCMC algorithm


## Setup
# True parameters
k0 = 2

# Model parameters
K = 20;  # number of components
gamma0 = 1/(2*K);  # Dirichlet concentration parameter
mu0 = 0.0;  # prior mean of component means
sigma0 = 5.0;  # prior standard deviation of component means
a0,b0 = 1.0,1.0;  # prior InverseGamma parameters for component variances

nreps = 1;  # number of times to run the simulation
n_total = 10^4;  # total number of MCMC iterations
n_burn = round(Int,n_total/10);  # number of MCMC iterations to discard as burn-in
n_init = round(Int,n_burn/2);  # number of MCMC iterations for initialization with periodic random splits
cutoff = 0.02;  # nonnegligible cluster size: 100*cutoff %


# Scenario setup
relative_size = "big-small"
relative_mis = "big-small"
close = false

# Run coarsening _________________________________________________________________________________________________


alpha = 631
## Run coarsening nreps times for each dataset
ns = [10000]
use = (n_burn+1:n_total)
n_use = length(use)
k_posteriors = zeros(K+1,nreps)
for (i_n,n) in enumerate(ns)
    for rep in 1:nreps
        println("n=$n  rep=$rep")
        dat = read_json("datasets/n=$n-close-$close-rsize-$relative_size-rmis-$relative_mis.json");
        x = dat["x_t"]
        # -------------------------- Coarsened mixture posterior --------------------------
        println("Running coarsened mixture...")
        zeta = alpha/(alpha + n)
        tic()
        N_c,p_c,mu_c,sigma_c,logw,L_c, z_c = sampler(x,n_total,n_init,K,gamma0,mu0,sigma0,a0,b0,zeta; mode="subset")
        toc()
        # compute weights for importance sampling
        w = exp.(logw[use] - logsumexp(logw[use]))
        cv = std(n_use*w)  # estimate coef of variation
        ess = 1/(1+cv^2)  # effective sample size
        @printf "ESS = %.3f%%\n" 100*ess

        kb_r = vec(sum(N_c.>n*zeta*cutoff, 1))
        pk,~ = histogram(kb_r[use],-0.5:1:K+0.5; weights=w)
        k_posteriors[:,rep,i_n] = pk

        save_res("results/coarsening-n=$n-close-$close-rsize-$relative_size-rmis-$relative_mis.json", N_c, mu_c, sigma_c, p_c, z_c, L_c)
    end
end
