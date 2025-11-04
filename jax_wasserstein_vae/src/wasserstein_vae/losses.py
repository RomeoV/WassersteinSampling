"""Loss functions for Wasserstein VAE."""

from typing import Callable, Optional
import jax
import jax.numpy as jnp
from scipy.stats import norm


def soft_histogram(values: jnp.ndarray, bin_centers: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """
    Differentiable soft histogram using Gaussian kernels.

    Args:
        values: Vector of values to bin, shape (N,)
        bin_centers: Vector of bin centers, shape (B,)
        sigma: Smoothness parameter

    Returns:
        Normalized soft histogram, shape (B,)
    """
    # Compute distances (N × B matrix)
    delta = values[:, None] - bin_centers[None, :]

    # Gaussian kernel weights
    weights = jnp.exp(-0.5 * (delta / sigma) ** 2)
    weights = weights / (jnp.sum(weights, axis=1, keepdims=True) + 1e-8)

    # Sum over samples and normalize
    histogram = jnp.sum(weights, axis=0)
    return histogram / jnp.sum(histogram)


def calibration_loss(predicted_probs: jnp.ndarray,
                    num_bins: int = 20,
                    sigma: Optional[float] = None) -> float:
    """
    Calibration loss based on PIT (Probability Integral Transform) uniformity.

    Args:
        predicted_probs: CDF values (should be uniform if calibrated), shape (N,)
        num_bins: Number of bins for histogram
        sigma: Smoothness parameter (default: 1/num_bins)

    Returns:
        KL divergence from uniform distribution
    """
    if sigma is None:
        sigma = 1.0 / num_bins

    bin_centers = jnp.linspace(0.0, 1.0, num_bins)

    # Compute soft histogram
    hist = soft_histogram(predicted_probs, bin_centers, sigma)

    # Target uniform distribution
    uniform = jnp.ones(num_bins) / num_bins

    # KL divergence (add small epsilon for numerical stability)
    loss = jnp.sum(uniform * jnp.log((uniform + 1e-8) / (hist + 1e-8)))

    return loss


def kl_divergence_loss(mu: jnp.ndarray, logvar: jnp.ndarray) -> float:
    """
    KL divergence between N(mu, var) and N(0, 1).

    Args:
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution

    Returns:
        KL divergence loss
    """
    return jnp.mean(0.5 * (-1.0 - logvar + mu**2 + jnp.exp(logvar)))


def huber_loss(x: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    """
    Huber loss (smooth approximation to absolute value).

    Args:
        x: Input values
        delta: Threshold for switching between quadratic and linear

    Returns:
        Huber loss values
    """
    abs_x = jnp.abs(x)
    return jnp.where(
        abs_x <= delta,
        0.5 * x**2,
        delta * (abs_x - 0.5 * delta)
    )


def soft_indicator(y_pred: jnp.ndarray, y_threshold: jnp.ndarray, temperature: float = 0.1) -> jnp.ndarray:
    """
    Soft indicator function using sigmoid.

    Approximates 1[y_pred > y_threshold] in a differentiable way.

    Args:
        y_pred: Predicted values, shape (..., )
        y_threshold: Threshold values, can be broadcasted
        temperature: Temperature for sigmoid (smaller = sharper transition)

    Returns:
        Soft indicator values in [0, 1]
    """
    return jax.nn.sigmoid((y_pred - y_threshold) / temperature)


class WassersteinLossWAux:
    """
    Wasserstein loss with auxiliary losses for VAE training.

    Combines:
    1. Wasserstein distance using projections and soft indicators
    2. KL divergence for VAE latent space
    3. Auxiliary Gaussian NLL loss with learned uncertainty weighting
    """

    def __init__(
        self,
        num_samples: int = 256,
        num_thresholds: int = 256,
        temperature: float = 0.1,
        reducer_fn: Callable = jnp.mean,
        reference_distribution: Optional[object] = None,
    ):
        """
        Initialize Wasserstein loss.

        Args:
            num_samples: Number of samples to generate from VAE
            num_thresholds: Number of threshold values for Wasserstein distance
            temperature: Temperature for soft indicator
            reducer_fn: Function to reduce loss (e.g., jnp.mean, jnp.sum)
            reference_distribution: Reference distribution for sampling thresholds
        """
        self.num_samples = num_samples
        self.num_thresholds = num_thresholds
        self.temperature = temperature
        self.reducer_fn = reducer_fn
        self.reference_distribution = reference_distribution

    def get_threshold_values(self, rng: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Sample threshold values for Wasserstein distance computation.

        Args:
            rng: Random number generator

        Returns:
            Threshold values, shape (num_thresholds, 1)
        """
        if self.reference_distribution is not None:
            # Sample from reference distribution if available
            if hasattr(self.reference_distribution, 'sample'):
                return self.reference_distribution.sample(rng, (self.num_thresholds, 1))
            else:
                # Fall back to standard normal
                return jax.random.normal(rng, (self.num_thresholds, 1))
        else:
            # Default to standard normal
            return jax.random.normal(rng, (self.num_thresholds, 1))

    def __call__(
        self,
        params,
        apply_fn,
        x_input: jnp.ndarray,
        y_true: jnp.ndarray,
        rng: jax.random.PRNGKey,
    ) -> tuple:
        """
        Compute Wasserstein loss with auxiliary losses.

        Args:
            params: Model parameters
            apply_fn: Model's apply function
            x_input: Input to VAE (typically zeros for unconditional generation)
            y_true: True samples from target distribution, shape (batch_size, output_dim)
            rng: Random number generator

        Returns:
            total_loss: Combined loss value
            stats: Dictionary of individual loss components
        """
        rng_threshold, rng_encode, *rng_samples = jax.random.split(rng, 2 + self.num_samples)

        # Encode to get latent distribution parameters
        _, mu, logvar = apply_fn(
            {'params': params},
            x_input,
            training=True,
            rngs={'reparameterize': rng_encode}
        )

        # KL divergence loss
        kl_loss = kl_divergence_loss(mu, logvar)

        # Generate multiple samples from VAE by sampling and decoding
        sigma = jnp.exp(0.5 * logvar)

        def sample_and_decode(rng_key):
            # Sample latent code
            eps = jax.random.normal(rng_key, mu.shape)
            z = mu + sigma * eps

            # Decode using the model's decode method
            # We need to be careful here - apply_fn expects the full forward pass
            # So we'll use the encoder to get z, then manually decode
            # Actually, let's just use the full forward pass with a fresh sample
            y_pred, _, _ = apply_fn(
                {'params': params},
                x_input,
                training=True,
                rngs={'reparameterize': rng_key}
            )
            return y_pred

        # Generate samples: shape (num_samples, batch_size, output_dim)
        y_samples = jnp.stack([sample_and_decode(rng) for rng in rng_samples], axis=0)

        # Compute auxiliary Gaussian NLL loss
        # y_samples: (num_samples, batch_size, output_dim)
        # y_true: (batch_size, output_dim)
        mu_empirical = jnp.mean(y_samples, axis=0)  # (batch_size, output_dim)
        var_empirical = jnp.var(y_samples, axis=0) + 1e-6  # (batch_size, output_dim)

        nll_loss = self.reducer_fn(
            0.5 * ((y_true - mu_empirical)**2 / var_empirical + jnp.log(var_empirical))
        )

        # Weight auxiliary loss with learned uncertainty
        if 'log_var_aux' in params:
            log_var_aux = params['log_var_aux']
            weighted_nll_loss = jnp.exp(-jnp.sum(log_var_aux)) * nll_loss + jnp.sum(log_var_aux)
        else:
            weighted_nll_loss = nll_loss

        # Compute Wasserstein distance
        # Get threshold values: (num_thresholds, output_dim)
        b_vals = self.get_threshold_values(rng_threshold)  # (num_thresholds, 1)

        # Compute soft indicators for samples
        # y_samples: (num_samples, batch_size, output_dim)
        # b_vals: (num_thresholds, output_dim)
        # We want to compute P(y_sample > b) for each threshold

        # Reshape for broadcasting
        # y_samples: (num_samples, batch_size, 1, output_dim)
        # b_vals: (1, 1, num_thresholds, output_dim)
        y_samples_exp = y_samples[:, :, None, :]
        b_vals_exp = b_vals[None, None, :, :]

        # Compute soft indicators: (num_samples, batch_size, num_thresholds, output_dim)
        indicators_x = soft_indicator(y_samples_exp, b_vals_exp, self.temperature)

        # Average over samples to get P(X > b): (batch_size, num_thresholds, output_dim)
        prob_x = jnp.mean(indicators_x, axis=0)

        # Compute soft indicators for true samples
        # y_true: (batch_size, 1, output_dim)
        # b_vals: (1, num_thresholds, output_dim)
        y_true_exp = y_true[:, None, :]
        prob_y = soft_indicator(y_true_exp, b_vals_exp[0, 0, :, :][None, :, :], self.temperature)

        # Wasserstein distance using Huber loss
        wasserstein_loss = self.reducer_fn(huber_loss(prob_x - prob_y))

        # Total loss
        total_loss = kl_loss + wasserstein_loss + weighted_nll_loss

        # Statistics for monitoring
        stats = {
            'kl_loss': kl_loss,
            'wasserstein_loss': wasserstein_loss,
            'nll_loss': nll_loss,
            'weighted_nll_loss': weighted_nll_loss,
            'total_loss': total_loss,
        }

        if 'log_var_aux' in params:
            stats['log_var_aux'] = jnp.sum(params['log_var_aux'])

        return total_loss, stats
