"""Example training script for Wasserstein VAE with mixture of Gaussians."""

import time
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../src')

from wasserstein_vae.model import VAEWithAuxLoss
from wasserstein_vae.losses import WassersteinLossWAux
from wasserstein_vae.train import create_train_state, train_step


class MixtureOfGaussians:
    """Simple mixture of Gaussians distribution (JAX-compatible)."""

    def __init__(self, means, stds, weights=None):
        """
        Initialize mixture of Gaussians.

        Args:
            means: List of means for each component
            stds: List of standard deviations for each component
            weights: Mixing weights (default: uniform)
        """
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
    """Main training loop."""
    # Configuration
    BATCH_SIZE = 1024
    NUM_LATENT_DIMS = 3
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 500
    SEED = 1

    # Define reference distribution (mixture of 2 Gaussians)
    reference_dist = MixtureOfGaussians(
        means=[-2.0, 3.0],
        stds=[2.0, 1.0]
    )

    print("Setting up model and training state...")

    # Initialize RNG
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
        num_samples=256,
        num_thresholds=256,
        temperature=tau,
        reference_distribution=reference_dist,
    )

    print(f"Training VAE with Wasserstein loss...")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Latent dims: {NUM_LATENT_DIMS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Temperature: {tau:.4f}")
    print()

    # Training loop
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        # Generate batch
        rng, data_rng = jax.random.split(rng)
        y_batch = reference_dist.sample(data_rng, (BATCH_SIZE, 1))
        x_batch = jnp.zeros_like(y_batch)  # Unconditional generation

        # Training step
        state, loss, stats = train_step(state, x_batch, y_batch, loss_fn)

        # Logging
        if epoch % 50 == 0:
            elapsed_time = time.time() - start_time
            throughput = (epoch * BATCH_SIZE) / elapsed_time

            print(f"Epoch {epoch:3d} | "
                  f"Loss: {loss:.6f} | "
                  f"KL: {stats['kl_loss']:.6f} | "
                  f"Wass: {stats['wasserstein_loss']:.6f} | "
                  f"NLL: {stats['nll_loss']:.6f} | "
                  f"Throughput: {throughput:.1f} samples/s")

            if 'log_var_aux' in stats:
                print(f"         log_var_aux: {stats['log_var_aux']:.4f}")

    print(f"\nTraining completed in {time.time() - start_time:.2f}s")

    # Generate samples and plot
    print("\nGenerating samples...")
    rng, sample_rng = jax.random.split(rng)

    # Sample from prior
    num_samples = 10000
    z_samples = jax.random.normal(sample_rng, (num_samples, NUM_LATENT_DIMS))

    # Decode
    samples = model.apply(
        {'params': state.params},
        z_samples,
        method=lambda m, z: m.decode(z)
    )

    # Convert to numpy for plotting
    samples = np.array(samples).flatten()

    # Plot comparison
    print("Creating plot...")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram of samples
    ax.hist(samples, bins=50, density=True, alpha=0.7, label='VAE Samples', color='C0')

    # Reference distribution
    x_range = np.linspace(samples.min(), samples.max(), 200)
    ax.plot(x_range, reference_dist.pdf(x_range),
            'C1-', linewidth=2, label='Reference Distribution')

    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('VAE Samples vs Reference Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('vae_samples.png', dpi=150)
    print(f"Plot saved to vae_samples.png")

    # Print statistics
    print(f"\nSample statistics:")
    print(f"  Mean: {samples.mean():.4f}")
    print(f"  Std: {samples.std():.4f}")

    # Compute true statistics
    true_samples = reference_dist.sample(jax.random.PRNGKey(42), (10000,))
    print(f"\nReference statistics:")
    print(f"  Mean: {true_samples.mean():.4f}")
    print(f"  Std: {true_samples.std():.4f}")


if __name__ == '__main__':
    main()
