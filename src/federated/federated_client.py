"""
Federated Learning Client using Flower (flwr).

Each client represents a pilot with local telemetry data (Minari dataset partition).
Trains locally and sends only gradients (weights) to the server, not raw data.

Usage:
    python src/federated/federated_client.py \
        --client-id 1 \
        --server-address localhost:8080 \
        --data-path data/processed/pilot_1.csv
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import flwr as fl
    FLWR_AVAILABLE = True
except ImportError:
    FLWR_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SimpleMLP(nn.Module):
    """Simple MLP for sensor data classification."""

    def __init__(self, input_size: int = 6, hidden_size: int = 64, output_size: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def generate_synthetic_pilot_data(
    pilot_id: int,
    n_samples: int = 500,
    n_sensors: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic telemetry data specific to a pilot's style.

    Each pilot has a different style (aggressive, smooth, balanced).
    """
    np.random.seed(pilot_id)
    t = np.arange(n_samples) / 50.0

    # Pilot styles
    if pilot_id == 1:  # Aggressive braking
        ax = -3.0 * np.exp(-0.05 * t) + 0.5 * np.random.randn(n_samples)
        ay = 2.0 * np.sin(0.03 * t) + 0.3 * np.random.randn(n_samples)
        maneuver_type = 1  # Braking
    elif pilot_id == 2:  # Smooth cornering
        ax = 0.3 * np.sin(0.02 * t) + 0.2 * np.random.randn(n_samples)
        ay = 1.5 * np.sin(0.025 * t) + 0.2 * np.random.randn(n_samples)
        maneuver_type = 2  # Cornering
    else:  # Acceleration
        ax = 2.5 * (1.0 - np.exp(-0.03 * t)) + 0.3 * np.random.randn(n_samples)
        ay = 0.5 * np.sin(0.02 * t) + 0.2 * np.random.randn(n_samples)
        maneuver_type = 0  # Acceleration

    az = 9.81 + 0.15 * np.random.randn(n_samples)
    gx = 0.1 * np.sin(0.02 * t) + 0.05 * np.random.randn(n_samples)
    gy = 0.25 * np.sin(0.025 * t) + 0.05 * np.random.randn(n_samples)
    gz = 0.05 * np.sin(0.015 * t) + 0.02 * np.random.randn(n_samples)

    X = np.column_stack([ax, ay, az, gx, gy, gz]).astype(np.float32)
    y = np.full(n_samples, maneuver_type, dtype=np.int64)

    return X, y


def load_data(
    client_id: int,
    data_path: Path = None,
    n_samples: int = 500,
) -> Tuple[DataLoader, DataLoader]:
    """Load local client data (synthetic or from file)."""
    if data_path and data_path.exists():
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int64)
    else:
        logger.info(f"Generating synthetic data for pilot {client_id}")
        X, y = generate_synthetic_pilot_data(client_id, n_samples=n_samples, n_sensors=6)

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    dataset = TensorDataset(X_tensor, y_tensor)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    logger.info(f"Data loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    return train_loader, test_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    device: str,
    lr: float = 0.001,
) -> float:
    """Train for one epoch locally."""
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
) -> Tuple[float, float]:
    """Evaluate model on local test set."""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    return avg_loss, accuracy


class FlowerClient(fl.client.NumPyClient):
    """Flower client for federated learning."""

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str,
    ):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.local_losses = []

    def get_parameters(self, config: Dict) -> list:
        """Return model weights as flattened numpy arrays."""
        return [val.cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters: list) -> None:
        """Set model weights from numpy arrays."""
        for param, new_val in zip(self.model.parameters(), parameters):
            param.data = torch.from_numpy(new_val)

    def fit(self, parameters: list, config: Dict) -> Tuple[list, int, Dict]:
        """Train locally for specified epochs."""
        self.set_parameters(parameters)

        lr = config.get("lr", 0.001)
        epochs = config.get("epochs", 5)

        for epoch in range(epochs):
            loss = train_epoch(self.model, self.train_loader, self.device, lr=lr)
            self.local_losses.append(loss)
            logger.info(f"Client {self.client_id} - Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

        return self.get_parameters({}), len(self.train_loader.dataset), {"loss": loss}

    def evaluate(self, parameters: list, config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate locally."""
        self.set_parameters(parameters)
        loss, accuracy = evaluate(self.model, self.test_loader, self.device)
        logger.info(f"Client {self.client_id} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy, "loss": loss}


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--client-id", type=int, required=True, help="Client ID (1, 2, or 3)")
    parser.add_argument("--server-address", default="localhost:8080", help="Server address")
    parser.add_argument("--data-path", type=Path, help="Path to local data CSV")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Client {args.client_id} starting on {device}")

    # Load local data
    train_loader, test_loader = load_data(args.client_id, data_path=args.data_path, n_samples=500)

    # Create model
    model = SimpleMLP(input_size=6, hidden_size=64, output_size=3)

    # Create Flower client
    client = FlowerClient(args.client_id, model, train_loader, test_loader, device)

    # Connect to server
    logger.info(f"Connecting to server at {args.server_address}...")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )

    logger.info(f"Client {args.client_id} training complete")

    # Save local loss history
    output_path = Path("outputs") / f"client_{args.client_id}_loss.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"local_losses": client.local_losses}, f, indent=2)
    logger.info(f"Local losses saved to {output_path}")


if __name__ == "__main__":
    main()
