import json
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import joblib
import os

class WeightsDataset(Dataset):
    """
    Dataset to load and flatten client weights from JSON/JSONL files.
    Extracts net.2.weight, net.2.bias, net.4.weight, net.4.bias in that exact order.
    Total output vector length is 970.
    """
    def __init__(self, data_path, scaler=None, is_train=True):
        self.data_path = data_path
        self.raw_data = []
        self.flattened_weights = []
        
        self.load_data()
        self.process_weights()
        
        # Normalization
        if is_train:
            if scaler is None:
                self.scaler = StandardScaler()
                self.scaled_weights = self.scaler.fit_transform(self.flattened_weights)
            else:
                self.scaler = scaler
                self.scaled_weights = self.scaler.transform(self.flattened_weights)
        else:
            if scaler is None:
                raise ValueError("Must provide a fitted scaler for validation/test sets.")
            self.scaler = scaler
            self.scaled_weights = self.scaler.transform(self.flattened_weights)

    def load_data(self):
        """Loads data handling a single JSON Array, JSON Lines, or a directory of JSON files."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset path {self.data_path} not found.")
            
        if os.path.isdir(self.data_path):
            for filename in sorted(os.listdir(self.data_path)):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.data_path, filename)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.raw_data.extend(data)
                        elif isinstance(data, dict):
                            self.raw_data.append(data)
        else:
            with open(self.data_path, 'r') as f:
                first_char = f.read(1)
                f.seek(0)
                
                if first_char == '[':
                    self.raw_data = json.load(f)
                else:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.raw_data.append(json.loads(line))

    def process_weights(self):
        """Flattens the nested weight arrays into 1D vectors."""
        for item in self.raw_data:
            weights_dict = item.get("weights", {})
            
            # Extract and flatten each parameter according to the exact order in MnistNet
            # 1. net.2.weight (16, 49)
            w2 = np.array(weights_dict.get("net.2.weight", [])).flatten()
            
            # 2. net.2.bias (16,)
            b2 = np.array(weights_dict.get("net.2.bias", [])).flatten()
            
            # 3. net.4.weight (10, 16)
            w4 = np.array(weights_dict.get("net.4.weight", [])).flatten()
            
            # 4. net.4.bias (10,)
            b4 = np.array(weights_dict.get("net.4.bias", [])).flatten()
            
            # Concatenate all to get a 970-d vector
            flat_vec = np.concatenate([w2, b2, w4, b4])
            
            # Sanity check dimension
            if len(flat_vec) != 970:
                print(f"Warning: Expected 970 parameters, but got {len(flat_vec)} for round {item.get('round')} client {item.get('client_id')}")
                
            self.flattened_weights.append(flat_vec)
            
        self.flattened_weights = np.array(self.flattened_weights, dtype=np.float32)

    def __len__(self):
        return len(self.scaled_weights)

    def __getitem__(self, idx):
        # Return PyTorch tensor
        return torch.tensor(self.scaled_weights[idx], dtype=torch.float32)

    def save_scaler(self, save_path):
        """Saves the fitted scaler using joblib for later use (e.g. at inference)."""
        joblib.dump(self.scaler, save_path)

    @classmethod
    def load_scaler(cls, load_path):
        """Loads a fitted scaler."""
        return joblib.load(load_path)
