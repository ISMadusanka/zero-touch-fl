import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

class VAE(nn.Module):
    def __init__(self, input_dim=970, latent_dim=64):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    # Mean Squared Error for reconstruction
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL Divergence
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def get_dataloaders(csv_path: str, batch_size: int = 32, test_size: float = 0.2):
    logger.info(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"File {csv_path} not found. Please run data collection first.")
        raise
    
    # Drop 'round' column, keep only weights
    if 'round' in df.columns:
        weights = df.drop(columns=['round']).values
    else:
        weights = df.values
        
    logger.info(f"Loaded {weights.shape[0]} total samples with {weights.shape[1]} features.")
    
    # Split into train and test (chronological: first 80% train, last 20% test)
    X_train, X_test = train_test_split(weights, test_size=test_size, shuffle=False)
    logger.info(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def main():
    parser = argparse.ArgumentParser(description="Train VAE on client weights")
    parser.add_argument("--data", type=str, default="data/client0_weights.csv", help="Path to weights CSV")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent space dimension")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint-name", type=str, default="vae_client0.pt", help="Name of the checkpoint file")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Ensure checkpoint directory exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)

    # Initialize model and optimizer
    model = VAE(input_dim=970, latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    start_epoch = 1
    
    # Load checkpoint if exists to resume training
    if os.path.exists(checkpoint_path):
        logger.info(f"Found existing checkpoint at {checkpoint_path}. Loading...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resuming training from epoch {start_epoch}")
    else:
        logger.info("No checkpoint found. Starting fresh training.")

    # Load data
    train_loader, test_loader = get_dataloaders(args.data, batch_size=args.batch_size)
    
    if start_epoch > args.epochs:
        logger.info(f"Model has already been trained for {args.epochs} epochs. Exiting.")
        return

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # Validation loop
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for (data,) in test_loader:
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                loss = loss_function(recon_batch, data, mu, logvar)
                test_loss += loss.item()
                
        avg_test_loss = test_loss / len(test_loader.dataset)
        
        logger.info(f"Epoch: {epoch}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
        
        # Save checkpoint after every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
        }, checkpoint_path)
        
    logger.info(f"Training completed. Model saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
