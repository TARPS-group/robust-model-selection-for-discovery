# Simulation to calibrate and clustering using coarsened mixture
# Input: simulated data from "/data"
# Output: calibration results to "/results", calibration curve to "/figures"
# ________________________________________________________________________________________________________
# Before running include("run-calibration.jl"), change current directory to current path
cd("/Users/jwli/Documents/GitHub/robust-mixture-model-selection/code/simulation/gaussian-mixture-model")

## loading packages and functions
using Distributions
using PyPlot
using JSON
# using TickTock

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

n_total = 10^4;  # total number of MCMC iterations
n_burn = round(Int,n_total/10);  # number of MCMC iterations to discard as burn-in
n_init = round(Int,n_burn/2);  # number of MCMC iterations for initialization with periodic random splits
cutoff = 0.02;  # nonnegligible cluster size: 100*cutoff %



# Scenario setup
relative_size = "equal"
relative_mis = "bbig-small"
close = false

# Calibration _________________________________________________________________________________________________

## Read data
n = 10000
dat = read_json("datasets/n=$n-close-$close-rsize-$relative_size-rmis-$relative_mis.json");
x = dat["x_t"]


alphas = [10 .^collect(1:0.2:5); Inf]  # values of alpha to consider
n_alphas = length(alphas)
El = zeros(n_alphas);  # posterior expectation of the log likelihood, for each alpha
Ek = zeros(n_alphas);  # posterior expectation of the number of non-negligible clusters, for each alpha
for (i_a,a) in enumerate(alphas)
    println("alpha = $a")
    zeta = (1/n)/(1/n + 1/a)
    tic()
    N_r,p_r,mu_r,sigma_r,logw,L_r = sampler(x,n_total,n_init,K,gamma0,mu0,sigma0,a0,b0,zeta; mode="downweight")
    toc()
    use = n_burn+1:n_total
    El[i_a] = mean(L_r[use])
    kb_r = vec(sum(N_r.>n*cutoff, 1))
    Ek[i_a] = mean(kb_r[use])
end

# Save calibration results
res = Dict("Ek" => Ek, "El" => El, "alphas" => alphas, "k0" => k0);
stringres = JSON.json(res);
open("results/calibration-coarsening-results-n=$n-close-$close-rsize-$relative_size-rmis-$relative_mis.json", "w") do f
   write(f, stringres)
end
