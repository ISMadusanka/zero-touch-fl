# Documentation: Client 0 Weights CSV

## File Details
- **Path:** `data/client0_weights.csv`
- **Shape:** 2000 rows × 971 columns (1 round column + 970 weight values)
- **Use Case:** Training a Variational Autoencoder (VAE) on client weight distributions.

## What it Contains

Each row is a snapshot of **Client 0's full model weights** after local training in one FL round.

| Column | Type | Description |
|--------|------|-------------|
| `round` | int | Training round number (1–2000) |
| `w_0` – `w_15` | float | `net.2.bias` — bias of first Linear layer (16 values) |
| `w_16` – `w_799` | float | `net.2.weight` — weights of Linear(49→16), flattened (784 values) |
| `w_800` – `w_809` | float | `net.4.bias` — bias of second Linear layer (10 values) |
| `w_810` – `w_969` | float | `net.4.weight` — weights of Linear(16→10), flattened (160 values) |

> **Note:** Keys are sorted alphabetically (`net.2.bias` < `net.2.weight` < `net.4.bias` < `net.4.weight`), which makes the column order deterministic across rounds.

## Loading with pandas

```python
import pandas as pd

# Load the full CSV
df = pd.read_csv("data/client0_weights.csv")

print(df.shape)          # (2000, 971)
print(df.columns[:5])    # ['round', 'w_0', 'w_1', 'w_2', 'w_3']

# Separate round column from weight data
rounds = df["round"]
weights = df.drop(columns=["round"])  # (2000, 970) — pure weight matrix

print(weights.shape)     # (2000, 970)
print(weights.dtypes)    # all float64

# Convert to numpy array for VAE training
import numpy as np
X = weights.values       # shape: (2000, 970)

# Or convert to PyTorch tensor
import torch
X_tensor = torch.tensor(X, dtype=torch.float32)  # shape: (2000, 970)
```

## Quick Sanity Checks

```python
# Verify no NaN/Inf values
assert not df.isnull().any().any(), "Found NaN values!"
assert np.isfinite(weights.values).all(), "Found Inf values!"

# Basic stats
print(weights.describe())  # min, max, mean, std per column
print(f"Global mean: {weights.values.mean():.6f}")
print(f"Global std:  {weights.values.std():.6f}")
```