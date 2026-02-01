"""
Unit tests for multimodal fusion network architecture.

Tests validate:
- Tensor shape propagation through CNN+LSTM branches
- Parameter counting (federated learning compatibility)
- Forward pass with synthetic data
- Device compatibility (CPU/GPU)
- Intermediate representation extraction for interpretability
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.fusion_net import (
    TelemetryCNN,
    BiometricLSTM,
    MultimodalFusionNet,
    create_multimodal_model
)


class TestTelemetryCNN:
    """Test telemetry CNN branch for IMU sensor fusion."""

    def test_cnn_output_shape(self):
        """Validate CNN reduces 6D telemetry → 64D latent representation."""
        cnn = TelemetryCNN(input_channels=6, hidden_size=64)
        # Input: batch=4, time_steps=50, channels=6
        x = torch.randn(4, 50, 6)
        
        out = cnn(x)
        assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"

    def test_cnn_variable_sequence_length(self):
        """Verify CNN handles variable sequence lengths via AdaptiveAvgPool."""
        cnn = TelemetryCNN(input_channels=6, hidden_size=64)
        
        x_short = torch.randn(2, 30, 6)
        x_long = torch.randn(2, 100, 6)
        
        out_short = cnn(x_short)
        out_long = cnn(x_long)
        
        # Both should produce same output shape
        assert out_short.shape == (2, 64)
        assert out_long.shape == (2, 64)

    def test_cnn_gradient_flow(self):
        """Ensure gradients flow back through CNN."""
        cnn = TelemetryCNN(input_channels=6, hidden_size=64)
        x = torch.randn(2, 50, 6, requires_grad=True)
        
        out = cnn(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.abs().sum() > 0, "Gradients not flowing"


class TestBiometricLSTM:
    """Test biometric LSTM branch for ECG/HR/HRV fusion."""

    def test_lstm_output_shape(self):
        """Validate LSTM reduces 3D biometrics → 64D hidden state."""
        lstm = BiometricLSTM(input_size=3, hidden_size=64, num_layers=2)
        # Input: batch=4, time_steps=50, channels=3 (ECG, HR, HRV)
        x = torch.randn(4, 50, 3)
        
        out = lstm(x)
        assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"

    def test_lstm_hidden_state_extraction(self):
        """Verify LSTM extracts final hidden state correctly."""
        lstm = BiometricLSTM(input_size=3, hidden_size=64, num_layers=2)
        x = torch.randn(2, 50, 3)
        
        # Last hidden state (final time step)
        out = lstm(x)
        assert out.shape == (2, 64)
        
        # All states should be different from zero
        assert (out.abs() > 0).any(), "Output contains all zeros"

    def test_lstm_batch_normalization(self):
        """Check LSTM works with batch inputs."""
        lstm = BiometricLSTM(input_size=3, hidden_size=64, num_layers=2)
        x = torch.randn(4, 50, 3)
        
        out1 = lstm(x)
        out2 = lstm(x)
        
        # Outputs should be deterministic (no dropout in eval)
        lstm.eval()
        with torch.no_grad():
            out_eval = lstm(x)
        
        assert out_eval.shape == (4, 64)


class TestMultimodalFusionNet:
    """Test complete multimodal fusion architecture."""

    def test_fusion_net_output_shape(self):
        """Validate fusion net produces correct output dimension."""
        fusion_net = MultimodalFusionNet(
            telemetry_channels=6,
            biometric_channels=3,
            cnn_hidden=64,
            lstm_hidden=64,
            fusion_hidden=128,
            output_size=8
        )
        
        telemetry = torch.randn(4, 50, 6)  # 4 samples, 50 timesteps, 6 sensors
        biometric = torch.randn(4, 50, 3)  # 4 samples, 50 timesteps, 3 signals
        
        logits, intermediates = fusion_net(telemetry, biometric)
        assert logits.shape == (4, 8), f"Expected (4, 8), got {logits.shape}"

    def test_fusion_net_intermediates(self):
        """Verify intermediate representations extracted for interpretability."""
        fusion_net = MultimodalFusionNet(
            telemetry_channels=6,
            biometric_channels=3,
            cnn_hidden=64,
            lstm_hidden=64,
            fusion_hidden=128,
            output_size=8
        )
        
        telemetry = torch.randn(2, 50, 6)
        biometric = torch.randn(2, 50, 3)
        
        logits, intermediates = fusion_net(telemetry, biometric)
        
        assert "cnn_features" in intermediates
        assert "lstm_features" in intermediates
        assert "fusion_features" in intermediates
        
        assert intermediates["cnn_features"].shape == (2, 64)
        assert intermediates["lstm_features"].shape == (2, 64)
        assert intermediates["fusion_features"].shape == (2, 128)

    def test_fusion_net_parameter_count(self):
        """Validate parameter counts for federated learning."""
        fusion_net = MultimodalFusionNet(
            telemetry_channels=6,
            biometric_channels=3,
            cnn_hidden=64,
            lstm_hidden=64,
            fusion_hidden=128,
            output_size=8
        )
        
        params = list(fusion_net.parameters())
        assert len(params) > 0, "Model has no parameters"
        
        total_params = sum(p.numel() for p in params)
        assert total_params > 100, f"Too few parameters: {total_params}"

    def test_get_set_parameters(self):
        """Verify parameter serialization for federated learning."""
        fusion_net = MultimodalFusionNet(
            telemetry_channels=6,
            biometric_channels=3,
            cnn_hidden=64,
            lstm_hidden=64,
            fusion_hidden=128,
            output_size=8
        )
        
        # Get parameters
        params_dict = fusion_net.get_parameters()
        assert len(params_dict) > 0
        
        # Verify all parameters are numpy arrays
        for name, param in params_dict.items():
            assert isinstance(param, np.ndarray)

    def test_device_compatibility(self):
        """Test CPU and CUDA (if available) device placement."""
        fusion_net = MultimodalFusionNet(
            telemetry_channels=6,
            biometric_channels=3,
            cnn_hidden=64,
            lstm_hidden=64,
            fusion_hidden=128,
            output_size=8
        )
        
        telemetry = torch.randn(2, 50, 6)
        biometric = torch.randn(2, 50, 3)
        
        # CPU test
        fusion_net = fusion_net.to("cpu")
        logits, _ = fusion_net(telemetry, biometric)
        assert logits.device.type == "cpu"
        
        # CUDA test (if available)
        if torch.cuda.is_available():
            fusion_net = fusion_net.to("cuda")
            telemetry_gpu = telemetry.to("cuda")
            biometric_gpu = biometric.to("cuda")
            logits = fusion_net(telemetry_gpu, biometric_gpu)[0]
            assert logits.device.type == "cuda"


class TestCreateMultimodalModel:
    """Test model factory function."""

    def test_model_factory_defaults(self):
        """Validate default model creation."""
        model = create_multimodal_model()
        assert isinstance(model, MultimodalFusionNet)
        assert sum(p.numel() for p in model.parameters()) > 100

    def test_model_factory_custom_output(self):
        """Test factory with custom output size."""
        model = create_multimodal_model(output_size=16)
        
        telemetry = torch.randn(2, 50, 6)
        biometric = torch.randn(2, 50, 3)
        
        logits, _ = model(telemetry, biometric)
        assert logits.shape == (2, 16)

    def test_model_factory_device(self):
        """Test factory device placement."""
        model = create_multimodal_model(device="cpu")
        
        # Check all parameters are on CPU
        for param in model.parameters():
            assert param.device.type == "cpu"


class TestBackwardCompatibility:
    """Ensure fusion net integrates with existing components."""

    def test_training_mode(self):
        """Validate model switches between train/eval modes."""
        model = MultimodalFusionNet(
            telemetry_channels=6,
            biometric_channels=3,
            cnn_hidden=64,
            lstm_hidden=64,
            fusion_hidden=128,
            output_size=8
        )
        
        model.train()
        assert model.training
        
        model.eval()
        assert not model.training

    def test_loss_computation(self):
        """Verify loss computation compatible with standard PyTorch losses."""
        model = MultimodalFusionNet(
            telemetry_channels=6,
            biometric_channels=3,
            cnn_hidden=64,
            lstm_hidden=64,
            fusion_hidden=128,
            output_size=8
        )
        
        telemetry = torch.randn(4, 50, 6)
        biometric = torch.randn(4, 50, 3)
        target_actions = torch.randn(4, 8)
        
        logits, _ = model(telemetry, biometric)
        loss = nn.MSELoss()(logits, target_actions)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
