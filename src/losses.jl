"""
Loss functions for Wasserstein VAE training.
"""

using Lux
using ChainRulesCore
using Statistics
using Tullio
using Random
using ConcreteStructs
using Distributions
using NNlib: sigmoid

"""
Differentiable soft histogram using Gaussian kernels.

# Arguments
- `values`: Vector of values to bin
- `bin_centers`: Vector of bin centers
- `σ`: Smoothness parameter
"""
function soft_histogram(values::AbstractVector{T}, bin_centers, σ) where {T}
    # Compute distances (N × B matrix)
    delta = values .- bin_centers'

    # Gaussian kernel weights
    weights = exp.(-T(0.5) .* (delta ./ σ).^2)
    weights = weights ./ (sum(weights, dims=2) .+ eps(T))

    # Sum over samples and normalize
    histogram = sum(weights, dims=1)[:]
    return histogram ./ sum(histogram)
end

"""
Calibration loss based on PIT (Probability Integral Transform) uniformity.

# Arguments
- `predicted_probs`: CDF values (should be uniform if calibrated)
- `num_bins`: Number of bins for histogram
- `σ`: Smoothness parameter for soft binning
"""
function calibration_loss(predicted_probs::AbstractVector{T}, num_bins=20; σ=T(1/num_bins)) where {T}
    bin_centers = @ignore_derivatives range(T(0), T(1), length=num_bins)

    # Compute soft histogram
    hist = soft_histogram(predicted_probs, bin_centers, σ)

    # Target uniform distribution
    uniform = ones_like(hist, T, num_bins) ./ num_bins

    # KL divergence (add small epsilon for numerical stability)
    loss = sum(uniform .* log.((uniform .+ eps(T)) ./ (hist .+ eps(T))))

    return loss
end

"""
Soft indicator function using sigmoid for differentiability.

# Arguments
- `yhat`: Predicted values
- `y`: True values
- `temperature`: Temperature parameter for sigmoid smoothness
"""
function soft_indicator(yhat, y; temperature=0.1)
    sigmoid.((y .- yhat) ./ temperature)
end

"""
Huber loss for robust regression.
"""
huberloss(x) = (abs2(x) <= 1 ? abs2(x)/2 : abs(x) - 1//2)

"""
Base loss function type.
"""
abstract type LossFunction end

"""
Weight the auxiliary loss based on learned uncertainty.
"""
weightauxloss(m::VAE, l, ps, st) = l
weightauxloss(m::VAEWithAuxLoss, l, ps, st) =
    exp(-sum(ps.log_var_aux)) * l + sum(ps.log_var_aux)

"""
Wasserstein loss with auxiliary Gaussian NLL loss.

# Fields
- `nsamples`: Number of samples to generate from VAE
- `nb`: Number of threshold values for Wasserstein distance
- `tau`: Temperature for sigmoid smoothing
- `reducerfn`: Function to reduce loss values (e.g., mean, sum)
"""
@kwdef @concrete struct WassersteinLossWAux <: LossFunction
    nsamples
    nb
    tau = 0.1f0
    reducerfn = mean
    reference_distribution = nothing
end

"""
Sample threshold values from the reference distribution.
"""
function getbvals(l::WassersteinLossWAux)
    if l.reference_distribution !== nothing
        return rand(l.reference_distribution, (l.nb, 1))
    else
        # Default to standard normal if no reference provided
        return randn(Float32, (l.nb, 1))
    end
end

"""
Compute Wasserstein loss with auxiliary loss.

# Arguments
- `model`: VAE or VAEWithAuxLoss model
- `ps`: Model parameters
- `st`: Model state
- `(xin, ytrue)`: Tuple of (input, target) data
"""
function (l::WassersteinLossWAux)(model, ps, st, (xin, ytrue))
    # Encode to latent space
    (; μ, logσ²), st = encode(model, xin, ps, st)

    # KL divergence loss for VAE
    kldivloss = l.reducerfn(
        @. (-1 + -logσ² + μ^2 + exp(logσ²)) / 2
    )

    # Sample from latent distribution and decode
    σz = @. exp(0.5f0 * logσ²)
    ysamples = map(1:l.nsamples) do _
        # Sample from latent distribution
        ε = randn_like(μ)
        z = @. μ + σz * ε

        # Decode
        y, st = decode(model, z, ps, st)
        y
    end |> cols->stack(cols; dims=2)  # (outdim, nsamples, batchsize)

    # Compute empirical mean and variance for auxiliary loss
    μtotal, vartotal = let
        μtotal = mean(ysamples; dims=2)
        vartotal = var(ysamples; dims=2, mean=μtotal)
        (μtotal, vartotal)
    end .|> xs->dropdims(xs; dims=2)

    # Gaussian negative log-likelihood loss
    nllloss = l.reducerfn(
        @. 1//2 * ((ytrue - μtotal)^2 / (vartotal + eps(Float32)) + log(vartotal + eps(Float32)))
    )
    weighted_nllloss = weightauxloss(model, nllloss, ps, st)

    # Sample threshold values for Wasserstein distance
    bvals = @ignore_derivatives let
        Lux.adapt(typeof(μ), getbvals(l))
    end
    bdirs = ones_like(ysamples, (size(ysamples, 1), l.nb))

    # Project samples and true values
    @tullio xproj[ib, isample, ibatch] := ysamples[idim, isample, ibatch] * bdirs[idim, ib]
    @tullio yproj[ib, ibatch] := ytrue[idim, ibatch] * bdirs[idim, ib]

    # Compute probabilities using soft indicators
    prob_x = mean(
        @. sigmoid((xproj - bvals) / l.tau);
        dims=2
    ) |> xs->dropdims(xs; dims=2)
    prob_y = @. sigmoid((yproj - bvals) / l.tau)

    # Wasserstein distance using Huber loss
    wassersteinloss = l.reducerfn(
        @. huberloss(prob_x - prob_y)
    )

    # Combine all losses
    totalloss = kldivloss + wassersteinloss + weighted_nllloss

    return totalloss, st, (; μ, logσ², ysamples, bvals, bdirs, kldivloss, nllloss, weighted_nllloss, wassersteinloss)
end
