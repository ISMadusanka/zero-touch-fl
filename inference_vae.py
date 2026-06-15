import torch
import numpy as np
import json
import os
import sys
from dotenv import load_dotenv

# Ensure we can import from the rest of the project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.vae import VAE
from data.weights_dataset import WeightsDataset
from agents.llm_client import create_llm_client

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

def decode_latent_vector(latent_vector, model, scaler):
    """
    Takes a 64-D latent vector (numpy array or list), passes it through the VAE Decoder,
    un-normalizes the output, and rebuilds the 970-parameter PyTorch weight dictionary.
    """
    if isinstance(latent_vector, list):
        latent_vector = np.array(latent_vector)
        
    tensor_vec = torch.tensor([latent_vector], dtype=torch.float32)
    with torch.no_grad():
        reconstructed = model.decode(tensor_vec)
        
    recon_unscaled = scaler.inverse_transform(reconstructed.numpy())[0]
    
    w2 = torch.tensor(recon_unscaled[0:784].reshape((16, 49)), dtype=torch.float32)
    b2 = torch.tensor(recon_unscaled[784:800], dtype=torch.float32)
    w4 = torch.tensor(recon_unscaled[800:960].reshape((10, 16)), dtype=torch.float32)
    b4 = torch.tensor(recon_unscaled[960:970], dtype=torch.float32)
    
    return {
        "net.2.weight": w2,
        "net.2.bias": b2,
        "net.4.weight": w4,
        "net.4.bias": b4
    }

if __name__ == "__main__":
    load_dotenv()
    
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
    mock_dir = 'data/mock_weights'
    sample_file = sorted([f for f in os.listdir(mock_dir) if f.endswith('.json')])[0]
    full_path = os.path.join(mock_dir, sample_file)
    
    with open(full_path, 'r') as f:
        sample_row = json.load(f)
        
    mock_weights_dict = sample_row.get("weights")
    
    print("Encoding Client Weights into 64-D Latent Space...")
    latent_state, mse = get_latent_vector(mock_weights_dict, model, scaler, return_reconstruction=False)
    print(f"Honest MSE Reconstruction Error: {mse:.4f}")
    
    print("\nInitializing LLM Client...")
    try:
        llm = create_llm_client(backend="openai", model="gpt-4o-mini")
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        llm = None
        
    system_prompt = """
    You are a Latent Space optimizer.
    You receive a 64-D Latent State. You must output ONLY a JSON object indicating how to modify this latent state.
    Format:
    {
        "action": "positive_shift",
        "magnitude": 0.5
    }
    Supported actions: random_noise, positive_shift, negative_shift
    """
    
    instructions = [
        "Add a very small amount of random noise to bypass anomaly detection.",
        "Add a massive amount of random noise to completely destroy the model.",
        "Add a moderate positive shift to all features."
    ]
    
    for i, instruction in enumerate(instructions):
        print(f"\n{'='*50}")
        print(f"TEST {i+1}: {instruction}")
        print(f"{'='*50}")
        
        user_msg = json.dumps({
            "current_latent_state": latent_state.tolist(),
            "instruction": instruction
        })
        
        action = "random_noise"
        magnitude = 0.5
        
        if llm:
            try:
                response = llm.call(system_prompt, user_msg)
                print(f"LLM Response: {json.dumps(response)}")
                action = response.get("action", "random_noise")
                magnitude = float(response.get("magnitude", 0.5))
            except Exception as e:
                print(f"[!] LLM Call Failed (Check your .env API key): {e}")
                print(f"Falling back to default {action} with mag {magnitude}")
        else:
            print("No LLM available, using default fallback.")
            
        print(f"\nApplying Action '{action}' (magnitude={magnitude}) to Latent Space...")
        latent_delta = np.zeros_like(latent_state)
        if action == "random_noise":
            latent_delta = np.random.normal(0, magnitude, size=latent_state.shape)
        elif action == "positive_shift":
            latent_delta = np.full_like(latent_state, magnitude)
        elif action == "negative_shift":
            latent_delta = np.full_like(latent_state, -magnitude)
            
        modified_latent_state = latent_state + latent_delta
        
        print("Decoding Modified Latent Space back to 970 parameters...")
        reconstructed_weights = decode_latent_vector(modified_latent_state, model, scaler)
        
        print("\nReconstructed Weights Shape Validation:")
        for key, tensor_val in reconstructed_weights.items():
            print(f"  {key}: {tensor_val.shape}")
            
        print("\n--- Value Logging (First 5 elements) ---")
        print(f"0. Original Input Weights:\n   {mock_weights_dict['net.2.weight'][0][:5]}")
        print(f"1. Encoded Weights (Latent Vector):\n   {latent_state[:5].tolist()}")
        print(f"2. Noised Weights (Modified Latent Vector):\n   {modified_latent_state[:5].tolist()}")
        print(f"3. Decoded Noised Weights (net.2.weight):\n   {reconstructed_weights['net.2.weight'].flatten()[:5].tolist()}")
        
    print("\nStandalone test completed successfully.")
