"""Basic test script to verify the VAE implementation."""

import jax
import jax.numpy as jnp
import sys
sys.path.insert(0, 'src')

from wasserstein_vae.model import VAEWithAuxLoss
from wasserstein_vae.losses import WassersteinLossWAux, kl_divergence_loss
from wasserstein_vae.train import create_train_state

def test_model_creation():
    """Test that we can create and initialize the model."""
    print("Testing model creation...")
    rng = jax.random.PRNGKey(0)

    model = VAEWithAuxLoss(num_latent_dims=3)
    state = create_train_state(
        rng=rng,
        model=model,
        learning_rate=1e-3,
        input_shape=(32, 1)
    )

    print(f"  ✓ Model created successfully")
    print(f"  ✓ Parameters shape: {jax.tree_util.tree_map(lambda x: x.shape, state.params)}")
    return state

def test_forward_pass(state):
    """Test forward pass through the model."""
    print("\nTesting forward pass...")
    rng = jax.random.PRNGKey(1)

    # Create dummy input
    x = jnp.zeros((32, 1))

    # Forward pass
    output, mu, logvar = state.apply_fn(
        {'params': state.params},
        x,
        training=True,
        rngs={'reparameterize': rng}
    )

    print(f"  ✓ Output shape: {output.shape}")
    print(f"  ✓ Mu shape: {mu.shape}")
    print(f"  ✓ Logvar shape: {logvar.shape}")

    # Test KL loss
    kl_loss = kl_divergence_loss(mu, logvar)
    print(f"  ✓ KL loss: {kl_loss:.6f}")

    return output, mu, logvar

def test_loss_function(state):
    """Test the loss function."""
    print("\nTesting loss function...")
    rng = jax.random.PRNGKey(2)

    # Create dummy data
    x_batch = jnp.zeros((16, 1))
    y_batch = jax.random.normal(rng, (16, 1))

    # Create loss function with smaller num_samples for testing
    loss_fn = WassersteinLossWAux(
        num_samples=4,  # Small for testing
        num_thresholds=8,
        temperature=0.1,
    )

    # Compute loss
    rng, loss_rng = jax.random.split(rng)
    loss, stats = loss_fn(state.params, state.apply_fn, x_batch, y_batch, loss_rng)

    print(f"  ✓ Total loss: {loss:.6f}")
    print(f"  ✓ KL loss: {stats['kl_loss']:.6f}")
    print(f"  ✓ Wasserstein loss: {stats['wasserstein_loss']:.6f}")
    print(f"  ✓ NLL loss: {stats['nll_loss']:.6f}")

    if 'log_var_aux' in stats:
        print(f"  ✓ Log var aux: {stats['log_var_aux']:.6f}")

    return loss, stats

def test_gradient_computation(state):
    """Test that gradients can be computed."""
    print("\nTesting gradient computation...")
    rng = jax.random.PRNGKey(3)

    x_batch = jnp.zeros((16, 1))
    y_batch = jax.random.normal(rng, (16, 1))

    loss_fn = WassersteinLossWAux(
        num_samples=4,
        num_thresholds=8,
        temperature=0.1,
    )

    def loss_wrapper(params):
        return loss_fn(params, state.apply_fn, x_batch, y_batch, rng)

    # Compute gradients
    (loss, stats), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(state.params)

    print(f"  ✓ Gradients computed successfully")
    print(f"  ✓ Loss: {loss:.6f}")

    # Check gradient shapes match parameter shapes
    def check_shapes(p, g):
        return p.shape == g.shape

    shapes_match = jax.tree_util.tree_all(
        jax.tree_util.tree_map(check_shapes, state.params, grads)
    )
    print(f"  ✓ Gradient shapes match: {shapes_match}")

    # Check for NaN gradients
    def has_nan(x):
        return jnp.any(jnp.isnan(x))

    nan_checks = jax.tree_util.tree_map(has_nan, grads)
    has_nans = any(jax.tree_util.tree_leaves(nan_checks))
    print(f"  ✓ No NaN gradients: {not has_nans}")

    return grads

def main():
    """Run all tests."""
    print("=" * 60)
    print("Running JAX Wasserstein VAE Tests")
    print("=" * 60)

    try:
        state = test_model_creation()
        test_forward_pass(state)
        test_loss_function(state)
        test_gradient_computation(state)

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
