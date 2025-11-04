"""VAE model definition in JAX/Flax."""

from typing import Sequence, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn


class Encoder(nn.Module):
    """VAE Encoder that outputs latent distribution parameters."""

    num_latent_dims: int
    intermediate_dims: Sequence[int] = (20, 3)

    @nn.compact
    def __call__(self, x, training: bool = True):
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            training: Whether in training mode (for dropout, etc.)

        Returns:
            z: Sampled latent code
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        # Embedding layers
        h = x
        for dim in self.intermediate_dims:
            h = nn.Dense(dim)(h)
            h = nn.relu(h)

        # Project to mean and log variance
        mu = nn.Dense(self.num_latent_dims,
                     kernel_init=nn.initializers.xavier_uniform(),
                     bias_init=nn.initializers.zeros)(h)
        logvar = nn.Dense(self.num_latent_dims,
                         kernel_init=nn.initializers.xavier_uniform(),
                         bias_init=nn.initializers.zeros)(h)

        # Clamp log variance for stability
        logvar = jnp.clip(logvar, -20.0, 10.0)

        # Reparameterization trick
        if training:
            key = self.make_rng('reparameterize')
            eps = jax.random.normal(key, mu.shape)
            sigma = jnp.exp(0.5 * logvar)
            z = mu + sigma * eps
        else:
            # Use mean during inference
            z = mu

        return z, mu, logvar


class Decoder(nn.Module):
    """VAE Decoder that reconstructs from latent space."""

    output_dim: int = 1
    intermediate_dims: Sequence[int] = (20, 20)

    @nn.compact
    def __call__(self, z):
        """
        Decode latent code to output.

        Args:
            z: Latent code of shape (batch_size, latent_dim)

        Returns:
            x_rec: Reconstructed output
        """
        h = z
        for dim in self.intermediate_dims:
            h = nn.Dense(dim)(h)
            h = nn.relu(h)

        # Final output layer
        x_rec = nn.Dense(self.output_dim)(h)

        return x_rec


class VAE(nn.Module):
    """Complete VAE model."""

    num_latent_dims: int = 3
    encoder_intermediate_dims: Sequence[int] = (20, 3)
    decoder_intermediate_dims: Sequence[int] = (20, 20)
    output_dim: int = 1

    def setup(self):
        self.encoder = Encoder(
            num_latent_dims=self.num_latent_dims,
            intermediate_dims=self.encoder_intermediate_dims
        )
        self.decoder = Decoder(
            output_dim=self.output_dim,
            intermediate_dims=self.decoder_intermediate_dims
        )

    def __call__(self, x, training: bool = True):
        """
        Forward pass through VAE.

        Args:
            x: Input tensor
            training: Whether in training mode

        Returns:
            x_rec: Reconstructed output
            mu: Latent mean
            logvar: Latent log variance
        """
        z, mu, logvar = self.encoder(x, training=training)
        x_rec = self.decoder(z)
        return x_rec, mu, logvar

    def encode(self, x, training: bool = True):
        """Encode input to latent space."""
        return self.encoder(x, training=training)

    def decode(self, z):
        """Decode from latent space."""
        return self.decoder(z)

    def sample(self, rng, num_samples: int):
        """
        Sample from the prior and decode.

        Args:
            rng: Random number generator
            num_samples: Number of samples to generate

        Returns:
            Generated samples
        """
        z = jax.random.normal(rng, (num_samples, self.num_latent_dims))
        return self.decoder(z)


class VAEWithAuxLoss(nn.Module):
    """VAE wrapper that includes auxiliary loss parameters."""

    num_latent_dims: int = 3
    encoder_intermediate_dims: Sequence[int] = (20, 3)
    decoder_intermediate_dims: Sequence[int] = (20, 20)
    output_dim: int = 1

    def setup(self):
        self.vae = VAE(
            num_latent_dims=self.num_latent_dims,
            encoder_intermediate_dims=self.encoder_intermediate_dims,
            decoder_intermediate_dims=self.decoder_intermediate_dims,
            output_dim=self.output_dim
        )
        # Learnable log variance for auxiliary loss weighting
        self.log_var_aux = self.param('log_var_aux',
                                       nn.initializers.zeros,
                                       (1,))

    def __call__(self, x, training: bool = True):
        """Forward pass through VAE."""
        return self.vae(x, training=training)

    def encode(self, x, training: bool = True):
        """Encode input to latent space."""
        return self.vae.encode(x, training=training)

    def decode(self, z):
        """Decode from latent space."""
        return self.vae.decode(z)

    def sample(self, rng, num_samples: int):
        """Sample from the prior and decode."""
        return self.vae.sample(rng, num_samples)
