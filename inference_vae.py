import torch
import numpy as np
import json
import os
import sys

# Ensure we can import from the rest of the project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.vae import VAE
from data.weights_dataset import WeightsDataset

def get_latent_vector(weights_dict, model, scaler, noise_sd=0.0, return_reconstruction=False):
    """
    Takes a dictionary of weights, applies optional noise, flattens them, 
    scales them, and uses the VAE encoder to return the 64-D latent vector and MSE.
    If return_reconstruction is True, also returns the un-normalized reconstructed weights.
    """
    w2 = np.array(weights_dict.get("net.2.weight", []))
    b2 = np.array(weights_dict.get("net.2.bias", []))
    w4 = np.array(weights_dict.get("net.4.weight", []))
    b4 = np.array(weights_dict.get("net.4.bias", []))
    
    # 1. Apply Gaussian noise to simulate compromised weights
    if noise_sd > 0.0:
        w2 = w2 + np.random.normal(loc=0.0, scale=noise_sd, size=w2.shape)
        b2 = b2 + np.random.normal(loc=0.0, scale=noise_sd, size=b2.shape)
        w4 = w4 + np.random.normal(loc=0.0, scale=noise_sd, size=w4.shape)
        b4 = b4 + np.random.normal(loc=0.0, scale=noise_sd, size=b4.shape)
        
    # 2. Flatten the weights exactly as done during training
    flat_vec = np.concatenate([w2.flatten(), b2.flatten(), w4.flatten(), b4.flatten()])
    
    if len(flat_vec) != 970:
        raise ValueError(f"Expected 970 parameters, but got {len(flat_vec)}")
        
    # 3. Normalize using the saved scaler
    normalized_vec = scaler.transform([flat_vec])
    tensor_vec = torch.tensor(normalized_vec, dtype=torch.float32)
    
    # 4. Pass through the encoder and compute error
    with torch.no_grad():
        mu, logvar = model.encode(tensor_vec)
        # Reconstruct to check error
        reconstructed = model.decode(mu)
        mse = torch.nn.functional.mse_loss(reconstructed, tensor_vec).item()
        
        if return_reconstruction:
            # Un-normalize the reconstruction back to its original raw scale
            recon_unscaled = scaler.inverse_transform(reconstructed.numpy())[0]
            return mu[0].numpy(), mse, recon_unscaled, flat_vec
        else:
            return mu[0].numpy(), mse

if __name__ == "__main__":
    model_path = 'checkpoints/vae_model.pth'
    scaler_path = 'checkpoints/vae_scaler.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Error: Missing model or scaler checkpoints in 'checkpoints/' directory.")
        print("Please run `python train_vae.py --data_path <path>` first.")
        exit(1)
        
    print("Loading VAE Model and Scaler...")
    model = VAE(input_dim=970, latent_dim=64)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    scaler = WeightsDataset.load_scaler(scaler_path)
    
    print("\nLoading weights from the mock dataset for testing...")
    mock_dir = '/Users/pasindumallawarachchi/Documents/NonAcadamic/zero-touch-fl/data/mock_weights'
    sample_file = sorted([f for f in os.listdir(mock_dir) if f.endswith('.json')])[0]
    full_path = os.path.join(mock_dir, sample_file)
    
    with open(full_path, 'r') as f:
        sample_row = json.load(f)
        
    mock_weights_dict = sample_row.get("weights")
    
    print(f"Testing noise scenarios on {sample_file}...")
    
    # Test different noise levels
    noise_levels = [0.0, 0.1, 1.0]
    
    for noise in noise_levels:
        print(f"\n=== SCENARIO: Noise SD = {noise} ===")
        # Get the 64-dimensional output, reconstruction error, and the raw reconstructed weights!
        latent_vector, mse, recon_weights, noisy_input_weights = get_latent_vector(
            mock_weights_dict, model, scaler, noise_sd=noise, return_reconstruction=True
        )
        
        print(f"Reconstruction Error (MSE): {mse:.4f}")
        print("\nInput Weights (first 5 parameters):")
        print(noisy_input_weights[:5])
        print("Reconstructed Weights (first 5 parameters):")
        print(recon_weights[:5])
        
        print("----------------")
