# Quick Start Guide

## Installation

```bash
cd jax_wasserstein_vae
uv sync
```

## Running the Demo

### Simple Training Demo (No Plotting)

```bash
cd examples
uv run python simple_demo.py
```

This will:
- Train a VAE on a mixture of 2 Gaussians for 100 epochs (~12 seconds)
- Print training progress
- Generate samples and compare statistics

Expected output:
```
Epoch 100/100 | Loss:  3.9775 | KL: 0.0015 | Wass: 0.1009 | NLL: 4.0215 | 8570.2 samples/s
```

### Full Training with Plotting

```bash
cd examples
uv run python train_vae.py
```

This will:
- Train for 500 epochs
- Create a comparison plot (`vae_samples.png`)

## Running Tests

```bash
uv run python test_basic.py
```

All tests should pass:
```
✓ All tests passed!
```

## Key Components

### Model
- `VAE`: Basic VAE with encoder/decoder
- `VAEWithAuxLoss`: VAE with learnable auxiliary loss weight

### Loss
- `WassersteinLossWAux`: Combines:
  - Wasserstein distance (soft indicators)
  - KL divergence (latent regularization)
  - Gaussian NLL (auxiliary loss with learned weighting)

### Training
- `create_train_state`: Initialize model and optimizer
- `train_step`: JIT-compiled training step (auto-differentiates)

## Customization

To train on your own distribution:

```python
# Define your distribution
class MyDistribution:
    def sample(self, rng, shape):
        # Must be JAX-compatible
        return jax.random.normal(rng, shape) * 2.0 + 1.0

    def std(self):
        return 2.0

# Use in training
reference_dist = MyDistribution()
tau = 0.1 * reference_dist.std()

loss_fn = WassersteinLossWAux(
    num_samples=128,
    num_thresholds=128,
    temperature=tau,
    reference_distribution=reference_dist,
)
```

## Performance Notes

- **First iteration is slow** due to JIT compilation (~10s)
- After compilation: ~8500 samples/second (batch size 1024, CPU)
- Use GPU for faster training:
  ```bash
  # Install JAX with CUDA support
  uv add "jax[cuda]"
  ```

## Hyperparameter Tuning

Key parameters to adjust:

1. **num_samples** (default: 32-256): More samples = better Wasserstein estimate, slower training
2. **num_thresholds** (default: 32-256): More thresholds = finer CDF approximation
3. **temperature** (default: 0.1 * std): Smaller = sharper indicators, may be less stable
4. **learning_rate** (default: 1e-3): Standard Adam learning rate
5. **num_latent_dims** (default: 3): Latent space dimensionality

## Troubleshooting

### Slow compilation
- Reduce `num_samples` and `num_thresholds`
- First run compiles the function, subsequent runs are fast

### Poor sample quality
- Increase `num_epochs` (try 500-1000)
- Increase `num_samples` and `num_thresholds`
- Adjust `temperature` based on data scale
- Check loss components are balanced

### NaN losses
- Reduce learning rate
- Check data normalization
- Increase temperature for stability
