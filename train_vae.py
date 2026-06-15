import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model.vae import VAE, vae_loss_function
from data.weights_dataset import WeightsDataset

def train_vae(data_path, epochs, batch_size, learning_rate, beta):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    print(f"Loading dataset from {data_path}...")
    try:
        dataset = WeightsDataset(data_path=data_path, is_train=True)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {data_path}. Please provide a valid path.")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Loaded {len(dataset)} weight samples.")
    
    # 2. Init Model
    model = VAE(input_dim=970, latent_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    os.makedirs('checkpoints', exist_ok=True)
    
    # Save the scaler so we can normalize future incoming weights exactly the same way during the simulation
    scaler_path = os.path.join('checkpoints', 'vae_scaler.pkl')
    dataset.save_scaler(scaler_path)
    print(f"Saved weight standard-scaler to {scaler_path}")
    
    # 3. Train Loop
    print(f"Starting training for {epochs} epochs...")
    model.train()
    
    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_recon = 0
        total_kld = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            batch_data = batch_data.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, mu, logvar = model(batch_data)
            
            # Compute loss
            loss, recon_loss, kld_loss = vae_loss_function(reconstructed, batch_data, mu, logvar, beta=beta)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld_loss.item()
            
        avg_loss = total_loss / len(dataset)
        avg_recon = total_recon / len(dataset)
        avg_kld = total_kld / len(dataset)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}] \t Total Loss: {avg_loss:.4f} \t Recon: {avg_recon:.4f} \t KLD: {avg_kld:.4f}")
            
    # 4. Save Model
    model_save_path = os.path.join('checkpoints', 'vae_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Training complete. VAE weights saved to {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the VAE on benign client weights")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSON/JSONL weights dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta coefficient for KL Divergence")
    
    args = parser.parse_args()
    train_vae(args.data_path, args.epochs, args.batch_size, args.lr, args.beta)
