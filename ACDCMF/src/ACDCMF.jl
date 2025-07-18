"""
A technique to determine the appropriate rank K of an NMF
"""
module ACDCMF
include("./utils/utils.jl")

export UniformApproximate
export KDEUniform
export KL_distance_from_standard_uniform
export generate_empirical_eps_sets, structurally_aware_loss
export componentwise_loss, stare_from_componentwise_loss
export sample_eps_poisson!, sample_eps_normal!

using Distributions
using KernelDensity
using LinearAlgebra

abstract type UniformApproximate end

"""
Distribution represented by a kernel density estimation over sample points
"""
struct KDEUniform{T<:AbstractFloat} <: UniformApproximate
  samples::Vector{T}
  estimated_distr::UnivariateKDE
  KDEUniform(samples::Vector{T}; save_sample=false) where {T} = begin
    return new{T}(save_sample ? samples : T[], kde(samples, boundary=(0, 1), kernel=Epanechnikov))
  end
end

"""
Compute the KL Divergence D_KL(d|Uniform(0,1))
"""
function KL_distance_from_standard_uniform(d::KDEUniform)
  distr = d.estimated_distr
  width = step(distr.x)
  density = distr.density
  losses = width .* density .* log.(density)
  valid_losses = losses[.!isnan.(losses)]
  return sum(valid_losses)
end

"""
Returns a vector whose sum is the structurally aware loss when rho=0
nysamples: how many times to empirically sample y, larger value leads to more accurate approximation
sample_eps: a function that samples eps according to the assumed model
"""
function componentwise_loss(X::Matrix{R}, W::Matrix{F}, H::Matrix{F};
  approx_type::Type{T}=KDEUniform, nysamples::Integer=1,
  sample_eps=sample_eps_poisson!, approxargs=()) where {T<:UniformApproximate,F<:AbstractFloat,R<:Real}

  empirical_eps = generate_empirical_eps_sets(X, W, H, approx_type; nysamples, approxargs, sample_eps)
  return dropdims(sum(KL_distance_from_standard_uniform.(empirical_eps); dims=1); dims=1)
end

"""
Compute the structurally aware loss from the output of the function componentwise_loss
"""
function stare_from_componentwise_loss(cwl, rho; lambda=0.01)
  K = length(cwl)
  return sum(max.(0, cwl .- rho)) + lambda * K
end

"""
componentwise_loss and stare_from_componentwise_loss folded into one function
"""
function structurally_aware_loss(X::Matrix{R}, W::Matrix{F}, H::Matrix{F}, rho::F;
  lambda::F=0.01, approx_type::Type{T}=KDEUniform,
  kwargs...) where {T<:UniformApproximate,F<:AbstractFloat,R<:Real}

  componentwise_loss = componentwise_loss(X, W, H; approx_type, kwargs)
  K = length(componentwise_loss)
  return sum(max.(0, componentwise_loss .- rho)) + lambda * K
end

"""
Vectorized structurally_aware_loss w.r.t. rho
"""
function structurally_aware_loss(X::Matrix{R}, W::Matrix{F}, H::Matrix{F}, rho::Vector{F};
  lambda::F=0.01, approx_type::Type{T}=KDEUniform,
  kwargs...) where {T<:UniformApproximate,F<:AbstractFloat,R<:Real}

  componentwise_loss = componentwise_loss(X, W, H; approx_type, kwargs)
  K = length(componentwise_loss)
  return [sum(max.(0, componentwise_loss .- rh)) + lambda * K for rh in rho]
end

function sample_eps_poisson!(x, W, h; nys, eps_conditional)
  sanity_check = v -> if isnan(v[1])
    return ones(length(v)) / length(v)
  else
    return v
  end
  sample_eps = (y, lambda) -> begin
    a = cdf(Poisson(lambda), y - 1)
    b = cdf(Poisson(lambda), y)
    return a < b ? Uniform(a, b) |> rand : b
  end

  Wdh = W * Diagonal(h)
  y_dist = Multinomial.(x, sanity_check.(normalize.(eachrow(Wdh), 1)))
  for _ in 1:nys
    ys = reduce(hcat, rand.(y_dist)) |> transpose # DxK matrix
    push!.(eps_conditional, sample_eps.(ys, Wdh))
  end
end

function sample_eps_normal!(sigmas::Matrix{Float64})
  return (x, W, h; nys, eps_conditional) -> begin
    Wdh = W * Diagonal(h)
    D, K = size(Wdh)
    @assert size(sigmas) == size(W) "Expected sigmas to have the same dim as W: $(size(W)), got $(size(sigmas))"

    mu = (m1, m2, s1, s2) -> (s1^-2 * m1 - s2^-2 * m2) / (s1^-2 + s2^-2)
    sig = (s1, s2) -> s1 * s2 / sqrt(s1^2 + s2^2)
    deconvolve_normal = (x, n1, n2) -> Normal(mu(n1.μ, n2.μ - x, n1.σ, n2.σ), sig(n1.σ, n2.σ))
    sample_eps = (y, mu, sig) -> cdf(Normal(mu, sig), y)

    for _ in 1:nys
      ys = Array{Float64}(undef, (D, K))
      for d in 1:D
        sigs = @view sigmas[d, :]
        cum_sig_sq = sum(sigs .^ 2)
        cum_mu = sum(@view Wdh[d, :])
        cum_x = x[d]
        for k in 1:K-1
          this_mu = Wdh[d, k]
          this_sig = sigs[k]
          cum_mu -= this_mu
          cum_sig_sq -= this_sig^2

          n_y = Normal(this_mu, this_sig)
          n_rest = Normal(cum_mu, sqrt(cum_sig_sq))
          y = rand(deconvolve_normal(cum_x, n_y, n_rest))

          ys[d, k] = y
          cum_x -= y
        end
        ys[d, K] = cum_x
      end
      push!.(eps_conditional, sample_eps.(ys, Wdh, sigmas))
    end
  end
end

"""
Internal function to sample eps, return a UniformApproximate struct, which can be used in divergence computation
"""
function generate_empirical_eps_sets(X::Matrix{R}, W::Matrix{F}, H::Matrix{F}, approx_type::Type{T};
  nysamples::Integer=20, approxargs=(), sample_eps=sample_eps_poisson!) where {T<:UniformApproximate,F<:AbstractFloat,R<:Real}

  # sanity checking
  D_X, N_X = size(X)
  D_W, K_W = size(W)
  K_H, N_H = size(H)
  @assert D_X == D_W "Dimension mismatch between X, W, H"
  @assert K_W == K_H "Dimension mismatch between X, W, H"
  @assert N_X == N_H "Dimension mismatch between X, W, H"

  D = D_X
  K = K_W
  N = N_X

  eps_conditional = reshape([Float64[] for _ in 1:D*K], (D, K))
  for n in 1:N
    x = @view X[:, n]
    h = @view H[:, n]
    sample_eps(x, W, h; nys=nysamples, eps_conditional) # this mutates eps_conditional
  end
  return approx_type.(eps_conditional; approxargs...)
end

end
