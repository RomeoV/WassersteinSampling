# Wasserstein VAE in JAX

A JAX/Flax implementation of a Variational Autoencoder (VAE) trained with Wasserstein distance and auxiliary losses, reimplemented from the Julia/Lux version.

## Overview

This implementation features:

- **VAE Architecture**: Encoder-decoder architecture with reparameterization trick
- **Wasserstein Loss**: Uses sliced Wasserstein distance with soft indicators (sigmoid-based)
- **Auxiliary Losses**:
  - KL divergence for latent space regularization
  - Gaussian NLL loss with learned uncertainty weighting
- **Differentiable CDF Estimation**: Using soft indicators with temperature scaling

## Installation

This project uses `uv` for package management. First, make sure you have `uv` installed:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project:

```bash
cd jax_wasserstein_vae
uv sync
```

**Note**: The project is pinned to Python 3.10-3.12. If you have issues with pip not being found, uv will automatically download and use the correct Python version specified in `.python-version` (3.11).

### Troubleshooting Installation

If you see errors like `No module named pip`, try:

```bash
# Clean the environment and reinstall
rm -rf .venv
uv sync
```

Or specify Python 3.11 explicitly:

```bash
uv sync --python 3.11
```

## Project Structure

```
jax_wasserstein_vae/
├── src/
│   └── wasserstein_vae/
│       ├── __init__.py
│       ├── model.py          # VAE architecture
│       ├── losses.py         # Wasserstein and auxiliary losses
│       └── train.py          # Training utilities
├── examples/
│   └── train_vae.py          # Example training script
├── pyproject.toml            # Project configuration
└── README.md
```

## Usage

### Basic Training

Run the example script to train a VAE on a mixture of Gaussians:

```bash
cd examples
uv run train_vae.py
```

This will:
1. Train a VAE to sample from a mixture of two Gaussians
2. Print training progress with loss components
3. Generate samples and create a comparison plot (`vae_samples.png`)

### Custom Training

Here's a minimal example:

```python
import jax
import jax.numpy as jnp
from wasserstein_vae.model import VAEWithAuxLoss
from wasserstein_vae.losses import WassersteinLossWAux
from wasserstein_vae.train import create_train_state, train_step

# Create model
model = VAEWithAuxLoss(num_latent_dims=3)

# Create training state
state = create_train_state(
    rng=jax.random.PRNGKey(0),
    model=model,
    learning_rate=1e-3,
    input_shape=(batch_size, 1)
)

# Create loss function
loss_fn = WassersteinLossWAux(
    num_samples=256,
    num_thresholds=256,
    temperature=0.1,
)

# Training loop
for epoch in range(num_epochs):
    x_batch = jnp.zeros((batch_size, 1))  # For unconditional generation
    y_batch = sample_from_target_distribution(batch_size)

    state, loss, stats = train_step(state, x_batch, y_batch, loss_fn)
```

## Implementation Details

### Wasserstein Loss

The Wasserstein loss is computed using:

1. **Multiple samples**: Generate many samples from the VAE for each input
2. **Threshold values**: Sample threshold points from the reference distribution
3. **Soft indicators**: Use sigmoid functions to approximate indicator functions:
   ```
   P(X > b) ≈ sigmoid((X - b) / τ)
   ```
4. **Huber loss**: Robust distance metric between CDFs

### Auxiliary Loss Weighting

The Gaussian NLL auxiliary loss is weighted using learned uncertainty:

```
weighted_loss = exp(-log_σ²) * NLL_loss + log_σ²
```

This automatically balances the auxiliary loss with the main Wasserstein loss.

### Key Differences from Julia Implementation

- JAX uses explicit RNG key management (vs. implicit in Julia)
- Flax modules use `@nn.compact` or `setup()` methods
- Training state is immutable (functional programming style)
- Loss functions are designed to work with `jax.jit` compilation

## Key Components

### Model (`model.py`)

- `Encoder`: Maps input to latent distribution parameters (μ, log σ²)
- `Decoder`: Maps latent code to output
- `VAE`: Complete VAE with encode/decode/sample methods
- `VAEWithAuxLoss`: VAE wrapper with learnable auxiliary loss weight

### Losses (`losses.py`)

- `WassersteinLossWAux`: Main loss combining Wasserstein + KL + NLL
- `soft_histogram`: Differentiable histogram using Gaussian kernels
- `calibration_loss`: PIT-based calibration loss
- `soft_indicator`: Differentiable indicator function

### Training (`train.py`)

- `TrainState`: Training state with parameters, optimizer, and RNG
- `train_step`: JIT-compiled single training step
- `create_train_state`: Initialize model and optimizer

## Configuration

Key hyperparameters in the example:

- `BATCH_SIZE = 1024`: Batch size for training
- `NUM_LATENT_DIMS = 3`: Dimensionality of latent space
- `LEARNING_RATE = 1e-3`: Adam learning rate
- `num_samples = 256`: Number of VAE samples per input
- `num_thresholds = 256`: Number of threshold values for Wasserstein
- `temperature = 0.1 * std(data)`: Temperature for soft indicators

## Reference

This implementation is based on the Wasserstein sampling approach described in the Julia notebook `with_isaac_no_pkg.jl`, which trains a VAE to sample from arbitrary distributions using Wasserstein distance.

## License

MIT
