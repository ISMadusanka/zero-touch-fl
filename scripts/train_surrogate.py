import numpy as np
import pandas as pd
import xgboost as xgb
import os
import json

MODEL_PATH = "storage/surrogate_xgb.json"

def generate_synthetic_data(n_samples=500):
    """
    Generates synthetic data matching the 4-layer defense logic.
    Features: fl_trust, cluster_id, clipping_ratio, is_trimmed, raw_norm
    Label: 0 (Benign), 1 (Malicious)
    """
    np.random.seed(42)
    
    data = []
    for _ in range(n_samples):
        is_malicious = np.random.choice([0, 1], p=[0.7, 0.3])
        
        if is_malicious:
            # Malicious profiles: low trust, high chance of being trimmed/clustered
            fl_trust = np.random.uniform(0.0, 0.15)
            cluster_id = np.random.choice([0, 1], p=[0.3, 0.7])
            clipping_ratio = np.random.uniform(0.5, 0.95)
            is_trimmed = np.random.choice([0, 1], p=[0.2, 0.8])
            raw_norm = np.random.uniform(0.15, 0.5)
        else:
            # Benign profiles: high trust, low chance of being trimmed/clustered
            fl_trust = np.random.uniform(0.2, 1.0)
            cluster_id = np.random.choice([0, 1], p=[0.9, 0.1])
            clipping_ratio = np.random.uniform(0.95, 1.0)
            is_trimmed = np.random.choice([0, 1], p=[0.9, 0.1])
            raw_norm = np.random.uniform(0.05, 0.2)
            
        data.append({
            "layer_1_fl_trust": fl_trust,
            "layer_2_cluster": cluster_id,
            "layer_3_clipping": clipping_ratio,
            "layer_4_is_trimmed": is_trimmed,
            "raw_norm": raw_norm,
            "label": is_malicious
        })
        
    return pd.DataFrame(data)

def train_initial_model():
    print("Generating synthetic training data for surrogate...")
    df = generate_synthetic_data()
    
    X = df.drop("label", axis=1)
    y = df["label"]
    
    print("Training XGBoost surrogate model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective="binary:logistic",
        random_state=42
    )
    
    model.fit(X, y)
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_initial_model()
