"""
Example script: Training a simple Transformer-based model for maneuver detection.

This demonstrates how to train a model that can later be used with the
explainability pipeline. Uses synthetic telemetry data.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.explainability import SimpleTransformerEncoder


def generate_training_data(n_episodes: int = 100, seq_len: int = 128, n_sensors: int = 6):
    """Generate synthetic telemetry with binary maneuver labels."""
    X, y = [], []
    for _ in range(n_episodes):
        t = np.arange(seq_len) / 50.0
        # Random maneuver type
        maneuver_type = np.random.randint(0, 3)

        if maneuver_type == 0:  # Acceleration
            ax = 3.0 * np.sin(0.03 * t)
            ay = 0.5 * np.sin(0.02 * t)
        elif maneuver_type == 1:  # Hard braking
            ax = -4.0 * np.exp(-0.05 * t)
            ay = 1.0 * np.sin(0.02 * t)
        else:  # Cornering
            ax = 0.5 * np.sin(0.02 * t)
            ay = 2.5 * np.sin(0.03 * t)

        az = 9.81 + 0.2 * np.random.randn(seq_len)
        gx = 0.1 * np.sin(0.02 * t)
        gy = 0.3 * np.sin(0.025 * t)
        gz = 0.05 * np.sin(0.015 * t)

        signal = np.column_stack([ax, ay, az, gx, gy, gz]).astype(np.float32)
        signal += 0.3 * np.random.randn(*signal.shape)

        X.append(signal)
        y.append(maneuver_type)

    return np.array(X), np.array(y)


def train_model(model, train_loader, device, epochs: int = 5, lr: float = 1e-3):
    """Simple training loop."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits, _ = model(X_batch)
            logits = logits.squeeze(-1)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Generating training data...")
    X, y = generate_training_data(n_episodes=200, seq_len=128, n_sensors=6)

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y).long()

    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    print("Creating Transformer model...")
    model = SimpleTransformerEncoder(input_size=6, hidden_size=64, num_heads=4, num_layers=2)

    print(f"Training on {device}...")
    train_model(model, train_loader, device, epochs=5, lr=1e-3)

    output_dir = PROJECT_ROOT / "models"
    output_dir.mkdir(exist_ok=True)
    checkpoint_path = output_dir / "transformer_maneuver_detector.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "SimpleTransformerEncoder",
            "input_size": 6,
        },
        checkpoint_path,
    )
    print(f"Saved model to {checkpoint_path}")


if __name__ == "__main__":
    main()
