"""Simple training demo without plotting dependencies."""

import time
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm

import sys
sys.path.insert(0, '../src')

from wasserstein_vae.model import VAEWithAuxLoss
from wasserstein_vae.losses import WassersteinLossWAux
from wasserstein_vae.train import create_train_state, train_step


class MixtureOfGaussians:
    """Simple mixture of Gaussians distribution (JAX-compatible)."""

    def __init__(self, means, stds, weights=None):
        self.means = jnp.array(means)
        self.stds = jnp.array(stds)
        self.n_components = len(means)

        if weights is None:
            self.weights = jnp.ones(self.n_components) / self.n_components
        else:
            self.weights = jnp.array(weights)
            self.weights /= self.weights.sum()

    def sample(self, rng, shape):
        """Sample from the mixture distribution (JAX-compatible)."""
        rng_component, rng_value = jax.random.split(rng)

        # Sample component indices
        components = jax.random.categorical(
            rng_component,
            jnp.log(self.weights),
            shape=shape
        )

        # Sample from selected components
        samples = jax.random.normal(rng_value, shape)
        samples = samples * self.stds[components] + self.means[components]

        return samples

    def pdf(self, x):
        """Compute PDF of the mixture."""
        pdf_val = np.zeros_like(x)
        for i in range(self.n_components):
            pdf_val += np.array(self.weights[i]) * norm.pdf(x, np.array(self.means[i]), np.array(self.stds[i]))
        return pdf_val

    def std(self):
        """Compute standard deviation of the mixture."""
        # Use sample-based estimate
        rng = jax.random.PRNGKey(0)
        samples = self.sample(rng, (100000,))
        return float(jnp.std(samples))


def main():
    """Simple training demo."""
    # Configuration
    BATCH_SIZE = 1024
    NUM_LATENT_DIMS = 3
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 100  # Shorter for demo
    SEED = 1

    print("Wasserstein VAE Training Demo")
    print("=" * 60)

    # Define reference distribution (mixture of 2 Gaussians)
    reference_dist = MixtureOfGaussians(
        means=[-2.0, 3.0],
        stds=[2.0, 1.0]
    )
    print(f"Reference distribution: Mixture of 2 Gaussians")
    print(f"  Component 1: N(-2.0, 2.0)")
    print(f"  Component 2: N(3.0, 1.0)")
    print()

    # Initialize
    rng = jax.random.PRNGKey(SEED)
    rng, init_rng = jax.random.split(rng)

    # Create model
    model = VAEWithAuxLoss(
        num_latent_dims=NUM_LATENT_DIMS,
        encoder_intermediate_dims=(20, 3),
        decoder_intermediate_dims=(20, 20),
        output_dim=1
    )

    # Create training state
    state = create_train_state(
        rng=init_rng,
        model=model,
        learning_rate=LEARNING_RATE,
        input_shape=(BATCH_SIZE, 1)
    )

    # Create loss function
    tau = 0.1 * reference_dist.std()
    loss_fn = WassersteinLossWAux(
        num_samples=32,  # Reduced for faster compilation
        num_thresholds=32,
        temperature=tau,
        reference_distribution=reference_dist,
    )

    print(f"Configuration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Latent dims: {NUM_LATENT_DIMS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Temperature: {tau:.4f}")
    print(f"  Num samples: 32")
    print(f"  Num thresholds: 32")
    print()
    print("Note: First iteration will be slow due to JIT compilation...")
    print()

    # Training loop
    print("Training...")
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        # Generate batch
        rng, data_rng = jax.random.split(rng)
        y_batch = reference_dist.sample(data_rng, (BATCH_SIZE, 1))
        x_batch = jnp.zeros_like(y_batch)

        # Training step
        state, loss, stats = train_step(state, x_batch, y_batch, loss_fn)

        # Logging
        if epoch == 1:
            print("✓ Compilation complete! Training...")
            print()

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            samples_per_sec = (epoch * BATCH_SIZE) / elapsed

            print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
                  f"Loss: {loss:7.4f} | "
                  f"KL: {stats['kl_loss']:6.4f} | "
                  f"Wass: {stats['wasserstein_loss']:6.4f} | "
                  f"NLL: {stats['nll_loss']:6.4f} | "
                  f"{samples_per_sec:7.1f} samples/s")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f}s")
    print()

    # Generate and evaluate samples
    print("Generating samples...")
    rng, sample_rng = jax.random.split(rng)

    num_samples = 10000
    z_samples = jax.random.normal(sample_rng, (num_samples, NUM_LATENT_DIMS))

    # Use the decode method directly
    samples = model.apply(
        {'params': state.params},
        z_samples,
        method=lambda m, z: m.decode(z)
    )

    samples = np.array(samples).flatten()

    # Compute statistics
    print("Results:")
    print("-" * 60)

    # VAE statistics
    vae_mean = samples.mean()
    vae_std = samples.std()
    print(f"VAE samples:")
    print(f"  Mean: {vae_mean:7.4f}")
    print(f"  Std:  {vae_std:7.4f}")

    # Reference statistics
    true_samples = reference_dist.sample(jax.random.PRNGKey(42), (10000,))
    true_mean = true_samples.mean()
    true_std = true_samples.std()
    print(f"\nReference distribution:")
    print(f"  Mean: {true_mean:7.4f}")
    print(f"  Std:  {true_std:7.4f}")

    # Compute error
    mean_error = abs(vae_mean - true_mean)
    std_error = abs(vae_std - true_std)
    print(f"\nErrors:")
    print(f"  Mean error: {mean_error:.4f}")
    print(f"  Std error:  {std_error:.4f}")

    print("=" * 60)
    print("✓ Demo completed successfully!")


if __name__ == '__main__':
    main()
