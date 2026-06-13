"""Variational Autoencoder (VAE) for learning benign weight distributions.

Architecture:
    Encoder: 970 → 512 → 256 → 128 → (μ:62, logvar:62)
    Decoder: 62 → 128 → 256 → 512 → 970

The VAE learns the distribution of honest client weights from FL training.
Poisoned weights will have high reconstruction error, enabling anomaly detection.
"""

import torch
import torch.nn as nn


class WeightVAE(nn.Module):
    """VAE that encodes FL client weights into a latent space and reconstructs them.

    Args:
        input_dim: Dimension of flattened weight vector (default: 970)
        latent_dim: Dimension of latent space (default: 62)
        hidden_dims: List of hidden layer dimensions for encoder/decoder
    """

    def __init__(
        self,
        input_dim: int = 970,
        latent_dim: int = 62,
        hidden_dims: list[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        # --- Encoder ---
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
            ])
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space: μ and log(σ²)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # --- Decoder ---
        decoder_layers = []
        reversed_dims = list(reversed(hidden_dims))
        in_dim = latent_dim
        for h_dim in reversed_dims:
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
            ])
            in_dim = h_dim
        # Final output layer (no activation — weights can be any value)
        decoder_layers.append(nn.Linear(reversed_dims[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, 1).

        This allows gradients to flow through the sampling step.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During evaluation, use the mean directly (no sampling)
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstructed weights.

        Args:
            z: Latent vector of shape (batch_size, latent_dim)

        Returns:
            Reconstructed weights of shape (batch_size, input_dim)
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode → reparameterize → decode.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            x_hat: Reconstructed input (batch_size, input_dim)
            mu: Latent mean (batch_size, latent_dim)
            logvar: Latent log variance (batch_size, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute VAE loss = Reconstruction Loss (MSE) + β × KL Divergence.

    Args:
        x: Original input
        x_hat: Reconstructed input
        mu: Latent mean
        logvar: Latent log variance
        beta: Weight for KL divergence term (β-VAE)

    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction (MSE) loss
        kl_loss: KL divergence loss
    """
    # Reconstruction loss: Mean Squared Error
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction="mean")

    # KL divergence: -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss
