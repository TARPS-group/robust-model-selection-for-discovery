include("../src/ACDCMF.jl")

using Distributions
using LinearAlgebra
using NMF
using DataFrames
using CSV
using JLD2
using MAT
using Clustering
using .ACDCMF
using .ACDCMF.Utils

using CairoMakie
CairoMakie.activate!(type="svg")

function rank_determination(X, ks; nmfargs=())
  results = Array{NMF.Result}(undef, length(ks))
  nmfargs = (; alg=:multdiv, nmfargs...) # default algorithm
  nmfargs = nmfargs.alg == :bssmf ? nmfargs : (; tol=1e-4, nmfargs...)
  for (i, k) in collect(enumerate(ks))
    result = threaded_nmf(Float64.(X), k; maxiter=200000, nmfargs...)
    results[i] = result
  end
  results
end

function cache_result_hyprunmix(; overwrite=false, nysamples=5,
  dataset="urban", nmf_algs=["cd"], nmfargs=(), filenameappend="")
  println("program start...")
  cache_name = "nys=$(nysamples)"

  println("start looping...")
  data = CSV.read("./data/hyprunmix/$(dataset)/data.csv", DataFrame)
  X = Matrix{Int}(data)
  Base.Filesystem.mkpath("./caches/hyprunmix/$(dataset)/$(cache_name)/")
  for nmf_alg in nmf_algs
    if isfile("./caches/hyprunmix/$(dataset)/$(cache_name)/cache-$(nmf_alg)-hyprunmix-urban.jld2") && !overwrite
      continue
    end

    ks = 1:9
    results = rank_determination(X / 1000, ks;
      nmfargs=(; alg=Symbol(nmf_alg), maxiter=50000, replicates=1, ncpu=1, nmfargs...))

    componentwise_losses = Vector{Vector{Float64}}(undef, length(results))
    Threads.@threads for i in eachindex(results)
      r = results[i]
      # computing optimal sigma
      D, K = size(r.W)
      N = size(r.H, 2)
      numparams_per_row = K + (K - 1) * N
      rmse = X - (r.W * r.H * 1000) .|>
             (x -> x^2) |>
             (x -> sum(x; dims=2)) .|>
             (x -> x / (size(X, 2) - numparams_per_row)) .|>
             sqrt
      sigmas = rmse * ones(K)' / sqrt(K)
      componentwise_losses[i] = componentwise_loss(X, r.W * 1000, r.H; nysamples, approxargs=(), sample_eps=sample_eps_normal!(fill(sigmas, i)))
    end
    jldsave("./caches/hyprunmix/$(dataset)/$(cache_name)/cache-$(nmf_alg)-hyprunmix-urban-$(filenameappend).jld2";
      results, componentwise_losses)
  end
end

const default_result_generation_synthetic() = (;
  cache_name_prepend="",
  rgen=(data_file_name, X, ks, nmf_alg, nmfargs) -> begin
    results = rank_determination(X, ks;
      nmfargs=(; alg=Symbol(nmf_alg), replicates=16, ncpu=16, simplex_W=true, nmfargs...))
    return results
  end
)
const from_cache_synthetic(in_cache_name, out_cache_name_prepend="") = (;
  cache_name_prepend=out_cache_name_prepend,
  rgen=(data_file_name, _, _, nmf_alg, _) -> begin
    file = load("../caches/cancer-synthetic/$(in_cache_name)/cache-$(nmf_alg)-$(data_file_name).jld2")
    results = [r for r in file["results"]]
    return results
  end
)

function cache_result_synthetic(; overwrite=false, nysamples=500,
  result_generation=default_result_generation_synthetic(), nmfargs=(), nmf_algs=[],
  componentwise_loss_method=componentwise_loss, outfilenameappend="")

  println("program start...")
  cancer_categories = Dict(
    "breast_custom" => "600-breast-custom-seed-1",
  )
  misspecification_type = Dict(
    "none" => "",
    "contaminated" => "-contamination-2",
    "overdispersed" => "-overdispersed-2.0",
    "perturbed" => "-perturbed-0.0025"
  )
  cache_name_prepend, rgen = result_generation
  cache_name = "$(cache_name_prepend)nys=$(nysamples)"
  nmf_algs = nmf_algs

  println("start looping...")
  Base.Filesystem.mkpath("./caches/cancer-synthetic/$(cache_name)/")
  for nmf_alg in nmf_algs, cancer in keys(cancer_categories)
    loadings = CSV.read("./data/cancer-synthetic/synthetic-$(cancer_categories[cancer])-GT-loadings.csv", DataFrame; header=0)
    nloadings = nrow(loadings)

    for misspec in keys(misspecification_type)
      println("alg: $(nmf_alg)\tcancer: $(cancer)\tmisspec: $(misspec)")
      if isfile("./caches/cancer-synthetic/$(cache_name)/cache-$(nmf_alg)-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2") && !overwrite
        continue
      end

      data = CSV.read("./data/cancer-synthetic/synthetic-$(cancer_categories[cancer])$(misspecification_type[misspec]).tsv", DataFrame; delim='\t')
      X = Matrix(data[:, 2:end])

      ks = 1:nloadings+3
      results = rgen("$(cancer_categories[cancer])$(misspecification_type[misspec])", X, ks, nmf_alg, nmfargs)
      componentwise_losses = Vector{Vector{Float64}}(undef, length(results))
      Threads.@threads for i in eachindex(results)
        r = results[i]
        componentwise_losses[i] = componentwise_loss_method(X, r.W, r.H; nysamples, approxargs=())
      end
      jldsave("./caches/cancer-synthetic/$(cache_name)/cache-$(nmf_alg)-$(cancer_categories[cancer])$(misspecification_type[misspec])$(outfilenameappend).jld2"; results, componentwise_losses)
    end
  end
end

function generate_plots_hyprunmix(; cache_name="nys=20-multiplier=1", dataset="urban",
  nmf_algs=["cd"], rhos=0:0.1:40, filenameappend="")
  println("program start...")
  signatures = CSV.read("./data/hyprunmix/$(dataset)/signatures.csv", DataFrame)
  loadings = CSV.read("./data/hyprunmix/$(dataset)/loadings.csv", DataFrame)
  abundance_gt = matread("./data/hyprunmix/$(dataset)/end5_groundTruth.mat")["A"]
  nloadings = nrow(loadings)

  data = CSV.read("./data/hyprunmix/$(dataset)/data.csv", DataFrame)
  X = Matrix{Int}(data) / 1000
  D, N = size(X)

  println("start looping...")
  w_metric1 = (w, w_gt) -> 1 - (normalize(w)' * normalize(w_gt)) |> (x -> isnan(x) ? 1.0 : x)
  w_metric2 = (w, w_gt) -> norm(w - w_gt) |> (x -> isnan(x) ? 1.0 : x)
  w_metric = (args...) -> 2w_metric1(args...) + w_metric2(args...)
  for nmf_alg in nmf_algs
    Base.Filesystem.mkpath("./plots/hyprunmix/$(dataset)/$(cache_name)/composite-$(nmf_alg)/pdf")
    Base.Filesystem.mkpath("./plots/hyprunmix/$(dataset)/$(cache_name)/composite-$(nmf_alg)/svg")

    # jldsave("../result-cache/rho-k-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2"; rhos, ks, losses, results)
    # data = CSV.read("../synthetic-data-2023/synthetic-$(cancer_categories[cancer])$(misspecification_type[misspec]).tsv", DataFrame; delim='\t')
    # X = Matrix(data[:, 2:end])

    file = load("./caches/hyprunmix/$(dataset)/$(cache_name)/cache-$(nmf_alg)-hyprunmix-$(dataset)$(filenameappend).jld2")
    results = [r for r in file["results"]]
    println([r.converged for r in results])
    componentwise_losses = file["componentwise_losses"]

    valid_results = filter(results) do r
      size(r.H)[1] <= nloadings
    end
    fig = Figure(size=(750, 750))

    rho_k_losses(fig[2, 1], componentwise_losses, rhos; rho_choice=0.5)
    # subfig_bubs, ax_bubs = bubbles(
    #   fig[2, 1],
    #   loadings,
    #   signatures,
    #   results;
    #   w_metric,
    #   colorrange=(0, 3),
    #   weighting_function=(wdiff, hdiff) -> wdiff
    # )
    # bubs_legend_and_colorbar = GridLayout()
    # bubs_legend_and_colorbar[1:2, 1] = subfig_bubs.content[2].content.content .|> x -> x.content


    k_labels = results .|> x -> size(x.H, 1)
    ax1 = Axis(fig[1, 1][2, 1]; yscale=identity, xticks=(1.5:length(results)+0.5, ["$(i)" for i in k_labels]),
      # yticks=0:length(results), 
      xlabel="K", ylabel=L"\text{BIC}(\times 10^7)", xlabelsize=20, ylabelsize=20)
    ax2 = Axis(fig[1, 1][1, 1]; yscale=identity, limits=(nothing, (0, nothing)),
      xticks=(1.5:length(results)+0.5, ["$(i)" for i in k_labels]), xlabel="K", ylabel="sARI", xlabelsize=20, ylabelsize=20)

    adjusted_rand_indices = [sARI(abundance_gt, results[k].H) for k in eachindex(results)]

    modelargs = [norm(r.W * r.H - X) / sqrt(D * N) for r in results] .|> x -> (x,)
    model = (m, s) -> Normal(m, s)
    # modelargs = [() for _ in 1:6]
    # model = x -> Poisson(x)
    bic = [BIC(X, results[k]; model, modelargs=modelargs[k]) for k in eachindex(results)] / 1e7
    bic_order = sortperm(bic) |> invperm
    lines!(ax1, 1.5:length(results)+0.5, bic; color=:red, label="BIC ranking")
    # lines!(ax1, 1.5:length(valid_results)+0.5, bic; label="BIC")

    lines!(ax2, 1.5:length(results)+0.5, [ari[1] for ari in adjusted_rand_indices]; color=:orange)
    vlines!(ax2, [6.5], color=:red, linestyle=:dash, label="BIC")
    vlines!(ax2, [4.5], color=:blue, linestyle=:dash, label="our method")
    linkxaxes!(ax1, ax2)
    axislegend(ax2; nbanks=2)
    # subfig_bubs[1, 1] = ax1
    # subfig_bubs[2, 1] = ax2
    # subfig_bubs[3, 1] = ax_bubs
    #
    #
    # subfig_bubs[1, 2] = Legend(fig, ax1)
    # subfig_bubs[2, 2] = Legend(fig, ax2)
    # subfig_bubs[3, 2] = bubs_legend_and_colorbar

    # rowsize!(subfig_bubs, 3, Relative(1 // 2))
    rowsize!(fig.layout, 2, Relative(1 // 3))

    # fig[0, :] = Label(fig, "Urban - Hyperspectral unmixing"; tellwidth=false, fontsize=30)

    save("./plots/hyprunmix/$(dataset)/$(cache_name)/composite-$(nmf_alg)/pdf/composite-$(nmf_alg)-$(dataset)$(filenameappend).pdf", fig)
    save("./plots/hyprunmix/$(dataset)/$(cache_name)/composite-$(nmf_alg)/svg/composite-$(nmf_alg)-$(dataset)$(filenameappend).svg", fig)
  end
end

function generate_plots_synthetic(; cache_name="nys=20-multiplier=200", nmf_algs=["multdiv"], rho_choice=Nothing, filenameappend="")
  println("program start...")
  signatures_unsorted = CSV.read("./data/cancer-synthetic/alexandrov2015_signatures.tsv", DataFrame; delim='\t')
  signatures = sort(signatures_unsorted)
  cancer_categories = Dict(
    "breast_custom" => "600-breast-custom-seed-1",
  )
  misspecification_type = Dict(
    "none" => "",
    "contaminated" => "-contamination-2",
    "overdispersed" => "-overdispersed-2.0",
    "perturbed" => "-perturbed-0.0025"
  )
  x_location = Dict(
    "none" => 0.8,
    "contaminated" => 0.8,
    "overdispersed" => 1.5,
    "perturbed" => 0.5
  )

  k_choice = Dict(
    "none" => 7,
    "contaminated" => 7,
    "overdispersed" => 7,
    "perturbed" => 8
  )

  println("start looping...")
  for nmf_alg in nmf_algs, cancer in keys(cancer_categories)
    loadings = CSV.read("./data/cancer-synthetic/synthetic-$(cancer_categories[cancer])-GT-loadings.csv", DataFrame; header=0)
    nloadings = nrow(loadings)
    Base.Filesystem.mkpath("./plots/synthetic/$(cache_name)/composite-$(nmf_alg)/pdf")
    Base.Filesystem.mkpath("./plots/synthetic/$(cache_name)/composite-$(nmf_alg)/svg")

    for misspec in keys(misspecification_type)
      println("alg: $(nmf_alg)\tcancer: $(cancer)\tmisspec: $(misspec)")
      # jldsave("../result-cache/rho-k-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2"; rhos, ks, losses, results)
      data = CSV.read("./data/cancer-synthetic/synthetic-$(cancer_categories[cancer])$(misspecification_type[misspec]).tsv", DataFrame; delim='\t')
      X = Matrix(data[:, 2:end])

      file = load("./caches/cancer-synthetic/$(cache_name)/cache-$(nmf_alg)-$(cancer_categories[cancer])$(misspecification_type[misspec])$(filenameappend).jld2")
      results = [r for r in file["results"]]
      componentwise_losses = file["componentwise_losses"]

      valid_results = filter(results) do r
        size(r.H)[1] <= nloadings
      end
      fig = Figure(size=(750, 750))
      rhos = 0:0.01:10

      rho_k_losses(fig[2, 1], componentwise_losses, rhos; rho_choice=x_location[misspec])
      # rho_k_bottom(fig[1, 1][2, 1], componentwise_losses)
      # subfig_bubs, ax_bubs = bubbles(fig[4, 1], loadings, signatures, results;
      #   weighting_function=(wdiff, hdiff) -> wdiff + tanh(0.1hdiff), simplex_W=true)
      # bubs_legend_and_colorbar = GridLayout()
      # bubs_legend_and_colorbar[1:2, 1] = subfig_bubs.content[2].content.content .|> x -> x.content


      k_labels = results .|> x -> size(x.H, 1)
      ax1 = Axis(fig[1, 1][1, 1]; yscale=log10, xticks=(1.5:length(results)+0.5, ["$(i)" for i in k_labels]),
        yaxisposition=:right, ytickcolor=:blue, yticklabelcolor=:blue, xlabel="K", ylabel=L"\mathrm{max} D_{rad}", ylabelsize=20)
      ax2 = Axis(fig[1, 1][1, 1]; yscale=identity, limits=(nothing, (0, nothing)),
        xticks=(1.5:length(results)+0.5, ["$(i)" for i in k_labels]),
        ytickcolor=:orange, yticklabelcolor=:orange, ylabel=L"\mathrm{max} D_{cos}", ylabelsize=20)
      ax3 = Axis(fig[1, 1][2, 1]; yscale=identity, xticks=(1.5:length(results)+0.5, ["$(i)" for i in k_labels]),
        xlabel="K", ylabel=L"\text{BIC}(\times 10^6)", ylabelsize=20)

      mrl_maxes = compare_against_gt.([loadings], [signatures], valid_results; weighting_function=(cd, ld) -> cd + tanh(0.1ld))
      plt1 = lines!(ax1, 1.5:length(valid_results)+0.5, [mm[1] for mm in mrl_maxes]; color=:blue)
      plt2 = lines!(ax2, 1.5:length(valid_results)+0.5, [mm[2] for mm in mrl_maxes]; color=:orange)

      modelargs = ()
      model = x -> Poisson(x)
      bic = [BIC(X, results[k]; model, modelargs) / 1e6 for k in eachindex(results)]
      bic_order = sortperm(bic) |> invperm
      plt3 = lines!(ax3, 1.5:length(results)+0.5, bic; color=:red)

      # linkxaxes!(ax_bubs, ax1, ax2, ax3)
      linkxaxes!(ax1, ax2, ax3)
      vlines!(ax1, [15.5], color=:red, linestyle=:dash, label="BIC")
      vlines!(ax1, [k_choice[misspec] + 0.5], color=:blue, linestyle=:dash, label="our method")
      axislegend(ax1; nbanks=2)
      # subfig_bubs[1, 1:2] = ax1
      # subfig_bubs[1, 1:2] = ax2
      # subfig_bubs[2, 1:2] = ax3
      # subfig_bubs[3, 1] = ax_bubs


      # subfig_bubs[1, 2] = Legend(fig, [plt1, plt2], ["max relative loading difference", "max cosine difference"])
      # subfig_bubs[2, 2] = Legend(fig, [plt3], ["BIC Order"])
      # axislegend(ax1, [plt1, plt2], [L"\mathrm{max} d_{rad}", L"\mathrm{max} D_{cos}"], labelsize=20)
      # axislegend(ax3, [plt3], ["BIC"], labelsize=20)
      # subfig_bubs[3, 2] = bubs_legend_and_colorbar

      rowsize!(fig.layout, 2, Relative(1 // 3))

      # fig[0, :] = Label(fig, "$(cancer_categories[cancer])$(misspecification_type[misspec])$(filenameappend)", tellwidth=false, fontsize=30)

      save("./plots/synthetic/$(cache_name)/composite-$(nmf_alg)/pdf/composite-$(nmf_alg)-$(cancer_categories[cancer])$(misspecification_type[misspec])$(filenameappend).pdf", fig)
      save("./plots/synthetic/$(cache_name)/composite-$(nmf_alg)/svg/composite-$(nmf_alg)-$(cancer_categories[cancer])$(misspecification_type[misspec])$(filenameappend).svg", fig)
    end
  end
end

# cache_result_synthetic(; overwrite=true, nmf_algs=["multdiv"], nysamples=500)
# generate_plots_synthetic(; cache_name="nys=500", nmf_algs=["multdiv"], rho_choice=0.9, filenameappend="")

# generate_plots_hyprunmix(; cache_name="nys=5", nmf_algs=["cd"], rhos=0:0.1:10, filenameappend="-l2h=m1e80-simplexh")
# cache_result_hyprunmix(; overwrite=true, nmf_algs=["cd"],
#   nmfargs=(; init=:nndsvda, replicates=16, ncpu=16, maxiter=10000, α=-1e80, regularization=:components, l₁ratio=0.0, simplex_H=true),
#   filenameappend="l2h=m1e80-simplexh")

