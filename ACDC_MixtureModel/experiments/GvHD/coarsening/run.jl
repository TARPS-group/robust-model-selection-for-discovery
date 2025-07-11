# Analysis of flow cytometry data using mixture model with standard MCMC.
# __________________________________________________________________________________________
# Settings

# Data settings
d = 7 # dataset id
varsubset = 3:6  # subset of the variables to use

# Model settings
K = 20  # number of components
gamma0 = 1/(2*K)  # Dirichlet concentration parameter

# Inference settings
n_total = 4000  # total number of MCMC iterations per run
n_burn = 2000  # number of MCMC iterations to discard as burn-in
n_init = round(Int,n_burn/5)  # number of MCMC iterations for initialization with periodic random splits
cutoff = 0  # nonnegligible cluster size: 100*cutoff %
from_file = false  # load previous results from file

# __________________________________________________________________________________________
# Supporting functions

using Distributions
using PyPlot
using ColorBrewer, Colors
using JSON


# set current path
cd("/Users/jwli/Documents/Bitbucket/structurally-aware-inference/code/experiments/FlowCytometry/")
# Code for mixture model MCMC algorithm
include("core.jl")
# Code for helper functions
include("helper.jl")

# __________________________________________________________________________________________
# Data

# Load the specified data set from the FlowCAP-I GvHD collection
function load_GvHD(d::Int)
    labels_file = "data/labels$(d).csv"
    data_file = "data/sample$(d).csv"
    labels = vec(readdlm(labels_file,',',Int))
    data,varnames = readdlm(data_file,',',Float64; header=true)
    return data,labels,varnames
end

# __________________________________________________________________________________________
# Visualize each data set using pairwise scatterplots

if true
    for d in [7,10]
        data,labels,varnames = load_GvHD(d)
        x = data[:,varsubset]
        labels += 1
        varnames = varnames[varsubset]
        L = unique(labels)
        for (i,j) in [(1,2),(2,3),(3,4)]
            figure(1,figsize=(5.5,5.5)); clf()
            for l in L
                plot(x[labels.==l,j],x[labels.==l,i],"o",color=T10colors[l],ms=0.6)
            end
            title("$(varnames[i]) \$\\mathrm{vs}\$ $(varnames[j])  \$\\mathrm{(Manual)}\$",fontsize=17)
            xticks(fontsize=10)
            yticks(fontsize=10)
            savefig("figures/mixcyto-clusters-d=$d-$(i)vs$(j)-manual.png",dpi=150)
        end
    end
end


# __________________________________________________________________________________________
# Run algorithm on test datasets


# Load data
data,labels,varnames = load_GvHD(d)
X = data[:,varsubset]'
n = size(X,2)
	

# inference setting
ns = [100, 500, 1000, 3000, 5000, 10000] 
ns = [100, 500, 800, 2000, 5000, 10000] 

nns = length(ns)
nreps = 5
k_posteriors = zeros(K+1,nreps,nns)

# Data-dependent hyperparameters
mu0,L0,nu0,V0 = hyperparameters(X)

# Run for standard posteriors
for (i_n,n) in enumerate(ns)
   for rep in 1:nreps
        println("n=$n  rep=$rep")
	x = X[:,1:n]

        srand(0) # reset RNG
        # Run MCMC sampler
        tic()
	zeta = 1.0 # for standard posterior
        N_r,p_r,mu_r,L_r,z_r,zmax_r = sampler(x,n_total,n_init,K,gamma0,mu0,L0,nu0,V0,zeta)
        toc()
        use = n_burn+1:n_total  # subset of samples to use
        n_use = length(use)

        # Compute the posterior on the number of nonnegligible clusters
        kp_r = map(N->sum(N.>n*cutoff), N_r)
        counts,~ = histogram(kp_r[use],-0.5:1:K+0.5)
        k_posterior = counts/n_use
        k_posteriors[:,rep,i_n] = k_posterior
        

    end
 end  

# save results for further use
res = Dict("k_posteriors" => k_posteriors);
stringres = JSON.json(res);
open("results/data$d-kposterior-standard.json", "w") do f
   write(f, stringres)
end

# read results
# res = read_json("results/data$d-kposterior-standard.json");
# kr_posteriors = res["k_posteriors"]



# Plot results
k0 = length(unique(labels))
colors = palette("Blues", 9)[3:9]

Kshow = 20
# Plot posterior
function plot_k_posteriors(fignum,titlestring)
    figure(fignum,figsize=(8,2.5)); clf()
    subplots_adjust(bottom=0.2)
    for (i_n,n) in enumerate(ns)
        kp = vec(mean(k_posteriors[1:Kshow+1,:,i_n],2))
	plot(0:Kshow,kp,color = rgb_sequence(colors[i_n]) ,mec="k",label=latex("n=$n"),ms=8,lw=2)
	# plot(0:Kshow,kp,"$(colors[i_n])$(shapes[i_n])-",mec="k",label=latex("n=$n"),ms=8,lw=2)
    end
    axvline(k0,ymin=0,ymax=1,linestyle= "dotted", color = "black", lw =2, label="ground truth")
    #plot!([k0], seriestype="vline",label = "ground truth")
    title(latex("\\mathrm{$titlestring} (k_0=$k0)"),fontsize=17)
    xlim(0,Kshow)
    ylim(0,1.0)
    xticks(0:Kshow)
    xlabel(latex("k"),fontsize=16)
    ylabel(latex("\\pi(k | \\mathrm{data})"),fontsize=16)
    legend(fontsize=12)
end
plot_k_posteriors(4,"Standard posterior")
savefig("figures/data$d-kposterior-standard.png",dpi=150)


nothing

