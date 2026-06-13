"""Train a VAE on benign FL client weights.

Usage:
    python train_vae.py
    python train_vae.py --config configs/vae.yaml

Loads weight JSON files from training rounds, trains the VAE to learn
the distribution of benign weights, and saves the model + normalization stats.
"""

import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader, random_split

from data.weight_dataset import WeightDataset
from model.weight_vae import WeightVAE, vae_loss

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging():
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/vae_training.log", mode="a", encoding="utf-8")
    stream_handler = logging.StreamHandler(
        open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[file_handler, stream_handler],
    )

logger = logging.getLogger("train_vae")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: WeightVAE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    beta: float,
    device: str,
) -> dict:
    """Train for one epoch. Returns average losses."""
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0

    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()

        x_hat, mu, logvar = model(batch)
        loss, recon, kl = vae_loss(batch, x_hat, mu, logvar, beta=beta)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon_loss": total_recon / n_batches,
        "kl_loss": total_kl / n_batches,
    }


@torch.no_grad()
def validate(
    model: WeightVAE,
    dataloader: DataLoader,
    beta: float,
    device: str,
) -> dict:
    """Validate the model. Returns average losses."""
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0

    for batch in dataloader:
        batch = batch.to(device)
        x_hat, mu, logvar = model(batch)
        loss, recon, kl = vae_loss(batch, x_hat, mu, logvar, beta=beta)

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon_loss": total_recon / n_batches,
        "kl_loss": total_kl / n_batches,
    }


def plot_losses(train_history: list, val_history: list, save_path: str):
    """Plot training and validation loss curves."""
    epochs = range(1, len(train_history) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Total loss
    axes[0].plot(epochs, [h["loss"] for h in train_history], label="Train", color="#2196F3")
    axes[0].plot(epochs, [h["loss"] for h in val_history], label="Val", color="#F44336")
    axes[0].set_title("Total Loss", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Reconstruction loss
    axes[1].plot(epochs, [h["recon_loss"] for h in train_history], label="Train", color="#2196F3")
    axes[1].plot(epochs, [h["recon_loss"] for h in val_history], label="Val", color="#F44336")
    axes[1].set_title("Reconstruction Loss (MSE)", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # KL divergence
    axes[2].plot(epochs, [h["kl_loss"] for h in train_history], label="Train", color="#2196F3")
    axes[2].plot(epochs, [h["kl_loss"] for h in val_history], label="Val", color="#F44336")
    axes[2].set_title("KL Divergence", fontsize=14)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("KL")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Loss curves saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train VAE on benign FL weights")
    parser.add_argument(
        "--config", default="configs/vae.yaml", help="Path to VAE config file"
    )
    args = parser.parse_args()

    setup_logging()
    logger.info("=" * 60)
    logger.info("VAE Training — Benign Weight Distribution Learning")
    logger.info("=" * 60)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)["vae"]

    logger.info(f"Config: {config}")

    device = config.get("device", "cpu")
    beta = config["beta"]

    # --- Load dataset ---
    dataset = WeightDataset(
        weights_dir=config["weights_dir"],
        normalize=True,
    )
    input_dim = dataset.get_input_dim()
    logger.info(f"Input dimension: {input_dim}")

    # --- Train/val split ---
    train_size = int(len(dataset) * config["train_split"])
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    logger.info(f"Train samples: {train_size}, Val samples: {val_size}")

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # --- Create model ---
    model = WeightVAE(
        input_dim=input_dim,
        latent_dim=config["latent_dim"],
        hidden_dims=config["hidden_dims"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"VAE model created — {total_params:,} parameters")
    logger.info(f"Architecture: {input_dim} → {config['hidden_dims']} → {config['latent_dim']} → {list(reversed(config['hidden_dims']))} → {input_dim}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # --- Training loop ---
    train_history = []
    val_history = []
    best_val_loss = float("inf")

    for epoch in range(1, config["epochs"] + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, beta, device)
        val_metrics = validate(model, val_loader, beta, device)

        train_history.append(train_metrics)
        val_history.append(val_metrics)

        # Log every 50 epochs (and first/last)
        if epoch % 50 == 0 or epoch == 1 or epoch == config["epochs"]:
            logger.info(
                f"Epoch {epoch:4d}/{config['epochs']} | "
                f"Train Loss: {train_metrics['loss']:.6f} "
                f"(recon: {train_metrics['recon_loss']:.6f}, kl: {train_metrics['kl_loss']:.6f}) | "
                f"Val Loss: {val_metrics['loss']:.6f} "
                f"(recon: {val_metrics['recon_loss']:.6f}, kl: {val_metrics['kl_loss']:.6f})"
            )

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            os.makedirs(os.path.dirname(config["checkpoint_path"]), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "input_dim": input_dim,
            }, config["checkpoint_path"])

    logger.info(f"Best validation loss: {best_val_loss:.6f}")

    # Save normalization stats separately (needed for inference)
    norm_stats = dataset.get_normalization_stats()
    norm_path = config["checkpoint_path"].replace(".pt", "_norm_stats.pt")
    torch.save(norm_stats, norm_path)
    logger.info(f"Normalization stats saved to {norm_path}")

    # --- Plot loss curves ---
    os.makedirs("logs/visualizations", exist_ok=True)
    plot_losses(train_history, val_history, "logs/visualizations/vae_training_loss.png")

    logger.info("=" * 60)
    logger.info("VAE TRAINING COMPLETE")
    logger.info(f"Model saved to: {config['checkpoint_path']}")
    logger.info(f"Norm stats saved to: {norm_path}")
    logger.info(f"Loss plot saved to: logs/visualizations/vae_training_loss.png")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
