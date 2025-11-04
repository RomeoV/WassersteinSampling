"""
WassersteinSampling.jl

A Julia package for training Variational Autoencoders (VAEs) with Wasserstein-based
calibration losses for probabilistic prediction.

This package implements various VAE architectures and loss functions designed to
produce well-calibrated probabilistic predictions by minimizing Wasserstein distances
and ensuring proper probability integral transform (PIT) properties.
"""
module WassersteinSampling

# Core dependencies
using Lux
using Random
using Statistics
using Distributions
using ConcreteStructs
using StaticArrays
using ChainRulesCore
using Tullio

# Re-export commonly used functions
export VAE, VAEWithAuxLoss
export encoder, decoder, encode, decode
export MyUnivariateMixtureModel, create_reference_distribution
export WassersteinLossWAux, calibration_loss, soft_histogram, soft_indicator, huberloss
export LossFunction

# Include submodules
include("distributions.jl")
include("models.jl")
include("losses.jl")

end # module WassersteinSampling
