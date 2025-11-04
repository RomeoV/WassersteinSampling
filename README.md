# WassersteinSampling.jl

A Julia package for training Variational Autoencoders (VAEs) with Wasserstein-based calibration losses for probabilistic prediction.

## Overview

This package implements VAE architectures and loss functions designed to produce well-calibrated probabilistic predictions by:
- Minimizing Wasserstein distances between predicted and true distributions
- Ensuring proper Probability Integral Transform (PIT) properties
- Using auxiliary losses for improved training stability

## Package Structure

```
WassersteinSampling/
├── Project.toml                 # Package dependencies
├── README.md                    # This file
├── src/
│   ├── WassersteinSampling.jl  # Main module
│   ├── models.jl               # VAE architectures (encoder, decoder)
│   ├── losses.jl               # Loss functions
│   └── distributions.jl        # Custom distributions
├── examples/
│   ├── train_wasserstein_vae.jl # Training example
│   └── visualize_results.jl     # Visualization script
└── with_isaac_no_pkg.jl        # Original Pluto notebook (archived)
```

## Installation

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Quick Start

### Training a Model

```julia
using WassersteinSampling
using Lux, Random, Optimisers

# Create reference distribution
reference_distribution = create_reference_distribution()

# Initialize model
rng = Random.Xoshiro(1)
vae = VAEWithAuxLoss(rng; num_latent_dims=3)
ps, st = Lux.setup(rng, vae)

# Create loss function
tau = 0.1f0 * std(reference_distribution)
lossfn = WassersteinLossWAux(;
    nsamples = 256,
    nb = 256,
    tau = tau,
    reference_distribution = reference_distribution
)

# Train (see examples/train_wasserstein_vae.jl for full example)
```

### Running Examples

Train a model:
```bash
julia --project=. examples/train_wasserstein_vae.jl
```

Visualize results (after training):
```bash
julia --project=. examples/visualize_results.jl
```

## Key Components

### Models

- `VAE`: Basic Variational Autoencoder with encoder and decoder
- `VAEWithAuxLoss`: VAE with learned auxiliary loss weighting
- `encoder()`: Encoder network mapping inputs to latent distributions
- `decoder()`: Decoder network mapping latent variables to outputs

### Loss Functions

- `WassersteinLossWAux`: Main loss combining:
  - Wasserstein distance via soft thresholding
  - KL divergence for VAE regularization
  - Auxiliary Gaussian NLL loss with learned weighting
- `calibration_loss()`: PIT-based calibration loss using soft histograms
- `soft_indicator()`: Differentiable indicator function using sigmoid

### Distributions

- `MyUnivariateMixtureModel`: Custom isbits-compatible mixture model
- `create_reference_distribution()`: Helper to create bimodal mixture

## Configuration

Key hyperparameters in `examples/train_wasserstein_vae.jl`:

- `batchsize`: Training batch size (default: 1024)
- `num_latent_dims`: Latent space dimensions (default: 3)
- `learning_rate`: Adam learning rate (default: 1e-3)
- `nsamples`: Number of VAE samples for loss (default: 256)
- `nb`: Number of Wasserstein thresholds (default: 256)
- `tau`: Temperature for soft indicators (default: 0.1 * std(data))

## Theory

The Wasserstein loss minimizes the 1-Wasserstein distance using:

$$\mathbb{E}_{b \sim p_y} |\mathbb{P}(X > b) - \mathbb{P}(Y > b)|$$

where:
- $X$ represents samples from the VAE
- $Y$ represents true samples
- $b$ are threshold values sampled from the reference distribution

The soft indicator makes this differentiable:

$$\sigma\left(\frac{y - \hat{y}}{\tau}\right)$$

See the original Pluto notebook (`with_isaac_no_pkg.jl`) for detailed derivations.

## Development Notes

This package was extracted from a Pluto notebook. The original notebook is preserved
as `with_isaac_no_pkg.jl` for reference but is no longer maintained.

Key changes from the notebook:
- Modular structure with separate files for models, losses, and distributions
- Removed Pluto-specific code (PlutoUI, reactive cells)
- Added proper exports and documentation
- Created standalone example scripts

## GPU Support

The package supports CUDA-accelerated training. It will automatically use GPU if
CUDA is available:

```julia
using LuxCUDA
const xdev = CUDA.functional() ? gpu_device() : cpu_device()
```

## Dependencies

Core dependencies:
- Lux.jl: Neural network framework
- Distributions.jl: Probability distributions
- Tullio.jl: Tensor operations
- Optimisers.jl: Optimization algorithms
- CairoMakie.jl: Visualization (examples only)

See `Project.toml` for complete list.

## License

[Add your license here]

## Citation

[Add citation information if applicable]
