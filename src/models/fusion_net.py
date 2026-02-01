"""
Multimodal Fusion Network: Combines IMU (telemetry) + Biometric signals.

Architecture:
- Branch 1 (Telemetry): CNN to extract spatial-temporal patterns from IMU
- Branch 2 (Biometrics): LSTM to model temporal HR/HRV/stress dynamics
- Fusion: Concatenate both branches
- Head: Dense layers to predict driver state

This enables the coaching agent to adapt feedback based on both:
1. Vehicle dynamics (what the bike is doing)
2. Driver physiology (what the pilot is experiencing)
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TelemetryCNN(nn.Module):
    """CNN branch for IMU/telemetry signal processing.

    Input: (batch, seq_len, 6) -> accelerometer + gyroscope
    Output: (batch, cnn_hidden_size)
    """

    def __init__(self, input_channels: int = 6, hidden_size: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(64, hidden_size, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 6) telemetry sequence

        Returns:
            (batch, hidden_size) aggregated CNN features
        """
        x = x.transpose(1, 2)  # (batch, 6, seq_len)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.global_pool(x)  # (batch, hidden_size, 1)
        x = x.squeeze(-1)  # (batch, hidden_size)

        return x


class BiometricLSTM(nn.Module):
    """LSTM branch for physiological signal processing.

    Input: (batch, seq_len, 3) -> HR, HRV, stress
    Output: (batch, lstm_hidden_size)
    """

    def __init__(self, input_size: int = 3, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 3) biometric sequence

        Returns:
            (batch, hidden_size) LSTM output from last timestep
        """
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers, batch, hidden_size)
        output = h_n[-1, :, :]  # Last layer, all batches
        return output


class MultimodalFusionNet(nn.Module):
    """Multimodal fusion network combining telemetry and biometrics.

    Two-branch architecture:
    - Telemetry branch (CNN)
    - Biometric branch (LSTM)
    - Concatenation + dense layers
    """

    def __init__(
        self,
        telemetry_channels: int = 6,
        biometric_channels: int = 3,
        cnn_hidden: int = 64,
        lstm_hidden: int = 64,
        fusion_hidden: int = 128,
        output_size: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.telemetry_cnn = TelemetryCNN(input_channels=telemetry_channels, hidden_size=cnn_hidden)
        self.biometric_lstm = BiometricLSTM(input_size=biometric_channels, hidden_size=lstm_hidden)

        fusion_input_size = cnn_hidden + lstm_hidden

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Linear(fusion_hidden, output_size)

        logger.info(f"MultimodalFusionNet initialized:")
        logger.info(f"  Telemetry CNN: {telemetry_channels} -> {cnn_hidden}")
        logger.info(f"  Biometric LSTM: {biometric_channels} -> {lstm_hidden}")
        logger.info(f"  Fusion: {fusion_input_size} -> {fusion_hidden} -> {output_size}")

    def forward(
        self,
        telemetry: torch.Tensor,
        biometrics: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass through dual-branch fusion network.

        Args:
            telemetry: (batch, seq_len, 6) IMU data
            biometrics: (batch, seq_len, 3) HR, HRV, stress

        Returns:
            (output logits, dict with intermediate features)
        """
        # Branch 1: Telemetry CNN
        cnn_features = self.telemetry_cnn(telemetry)

        # Branch 2: Biometric LSTM
        lstm_features = self.biometric_lstm(biometrics)

        # Fusion
        fused = torch.cat([cnn_features, lstm_features], dim=1)
        fusion_out = self.fusion(fused)

        # Output head
        logits = self.head(fusion_out)

        # Return logits and intermediate activations for interpretability
        intermediates = {
            "cnn_features": cnn_features,
            "lstm_features": lstm_features,
            "fusion_features": fusion_out,
        }

        return logits, intermediates

    def get_parameters(self) -> dict:
        """Return model parameters for federated learning."""
        return {
            name: param.data.cpu().numpy()
            for name, param in self.named_parameters()
        }

    def set_parameters(self, parameters: dict) -> None:
        """Set model parameters from dictionary."""
        for name, param in self.named_parameters():
            if name in parameters:
                param.data = torch.from_numpy(parameters[name])


def create_multimodal_model(
    output_size: int = 8,
    device: str = "cpu",
) -> MultimodalFusionNet:
    """Factory function to create and return a multimodal fusion model."""
    model = MultimodalFusionNet(
        telemetry_channels=6,
        biometric_channels=3,
        cnn_hidden=64,
        lstm_hidden=64,
        fusion_hidden=128,
        output_size=output_size,
        dropout=0.2,
    )
    model.to(device)
    return model


if __name__ == "__main__":
    # Demo: Create model and test forward pass
    model = create_multimodal_model(output_size=8)

    batch_size = 4
    seq_len = 128
    device = "cpu"

    telemetry = torch.randn(batch_size, seq_len, 6, device=device)
    biometrics = torch.randn(batch_size, seq_len, 3, device=device)

    output, intermediates = model(telemetry, biometrics)

    print(f"\nMultimodal Fusion Network Test")
    print(f"  Input shapes:")
    print(f"    Telemetry: {telemetry.shape}")
    print(f"    Biometrics: {biometrics.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  CNN features: {intermediates['cnn_features'].shape}")
    print(f"  LSTM features: {intermediates['lstm_features'].shape}")
    print(f"  Fusion features: {intermediates['fusion_features'].shape}")
    print("✓ Model test passed")
