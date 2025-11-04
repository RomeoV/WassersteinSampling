"""Wasserstein VAE implementation in JAX."""

from .model import VAE, VAEWithAuxLoss, Encoder, Decoder
from .losses import WassersteinLossWAux, soft_histogram, calibration_loss
from .train import train_step, create_train_state

__all__ = [
    "VAE",
    "VAEWithAuxLoss",
    "Encoder",
    "Decoder",
    "WassersteinLossWAux",
    "soft_histogram",
    "calibration_loss",
    "train_step",
    "create_train_state",
]
