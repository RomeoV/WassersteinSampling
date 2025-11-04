"""Training utilities for Wasserstein VAE."""

from typing import Callable, Any
from functools import partial
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


class TrainState(train_state.TrainState):
    """Training state with additional fields."""
    rng: jax.random.PRNGKey


def create_train_state(
    rng: jax.random.PRNGKey,
    model: Any,
    learning_rate: float = 1e-3,
    input_shape: tuple = (1, 1),
) -> TrainState:
    """
    Create initial training state.

    Args:
        rng: Random number generator
        model: Flax model
        learning_rate: Learning rate for optimizer
        input_shape: Shape of input for initialization

    Returns:
        Initial training state
    """
    # Initialize parameters
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    dummy_input = jnp.ones(input_shape)

    variables = model.init(
        {'params': init_rng, 'reparameterize': dropout_rng},
        dummy_input,
        training=True
    )
    params = variables['params']

    # Create optimizer
    tx = optax.adam(learning_rate)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        rng=rng,
    )


@partial(jax.jit, static_argnums=(3,))
def train_step(
    state: TrainState,
    x_batch: jnp.ndarray,
    y_batch: jnp.ndarray,
    loss_fn: Callable,
) -> tuple:
    """
    Single training step.

    Args:
        state: Current training state
        x_batch: Input batch
        y_batch: Target batch
        loss_fn: Loss function

    Returns:
        Updated state, loss value, and statistics
    """
    # Split RNG for this step
    rng, new_rng = jax.random.split(state.rng)

    def loss_wrapper(params):
        return loss_fn(params, state.apply_fn, x_batch, y_batch, rng)

    # Compute loss and gradients
    (loss, stats), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(state.params)

    # Update parameters
    state = state.apply_gradients(grads=grads)
    state = state.replace(rng=new_rng)

    return state, loss, stats


def train_epoch(
    state: TrainState,
    data_generator: Callable,
    loss_fn: Callable,
    num_batches: int,
) -> tuple:
    """
    Train for one epoch.

    Args:
        state: Current training state
        data_generator: Function that generates (x_batch, y_batch)
        loss_fn: Loss function
        num_batches: Number of batches in epoch

    Returns:
        Updated state and average loss
    """
    total_loss = 0.0
    all_stats = []

    for _ in range(num_batches):
        x_batch, y_batch = data_generator()
        state, loss, stats = train_step(state, x_batch, y_batch, loss_fn)
        total_loss += loss
        all_stats.append(stats)

    avg_loss = total_loss / num_batches

    # Average statistics
    avg_stats = {}
    for key in all_stats[0].keys():
        avg_stats[key] = sum(s[key] for s in all_stats) / len(all_stats)

    return state, avg_loss, avg_stats
