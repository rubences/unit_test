"""
Tests for Motion Style Transfer VAE
====================================

Comprehensive test suite for Variational Autoencoder (VAE) architecture,
style generation, interpolation, and telemetry reconstruction.

Author: GitHub Copilot for Competitive Motorcycle Racing
Date: 2026-01-17
"""

import pytest
import torch
import numpy as np
from src.generative.motion_vae import (
    MotionEncoder,
    MotionDecoder,
    MotionVAE,
    vae_loss,
    generate_aggressive_style,
    generate_smooth_style,
    interpolate_styles,
    VAETrainer
)


# ============================================================================
# TEST ENCODER
# ============================================================================

class TestMotionEncoder:
    """Test suite for MotionEncoder."""
    
    def test_encoder_initialization(self):
        """Test encoder is properly initialized with correct dimensions."""
        encoder = MotionEncoder(
            input_dim=3,
            hidden_dim=128,
            latent_dim=32,
            num_layers=2,
            dropout=0.2
        )
        
        assert encoder.input_dim == 3
        assert encoder.hidden_dim == 128
        assert encoder.latent_dim == 32
        assert encoder.num_layers == 2
        
        # Check parameters exist
        assert encoder.fc_mu is not None
        assert encoder.fc_logvar is not None
    
    def test_encoder_forward_pass(self):
        """Test encoder forward pass produces correct output shapes."""
        encoder = MotionEncoder(input_dim=3, hidden_dim=64, latent_dim=16)
        
        # Create dummy input: [batch=8, seq_len=250, features=3]
        x = torch.randn(8, 250, 3)
        
        # Forward pass
        mu, logvar = encoder(x)
        
        # Check output shapes
        assert mu.shape == (8, 16), f"Expected (8, 16), got {mu.shape}"
        assert logvar.shape == (8, 16), f"Expected (8, 16), got {logvar.shape}"
    
    def test_encoder_output_range(self):
        """Test encoder outputs reasonable values (no NaN or Inf)."""
        encoder = MotionEncoder(input_dim=3, hidden_dim=64, latent_dim=16)
        
        x = torch.randn(4, 250, 3)
        mu, logvar = encoder(x)
        
        # Check for NaN or Inf
        assert not torch.isnan(mu).any(), "Encoder mu contains NaN"
        assert not torch.isinf(mu).any(), "Encoder mu contains Inf"
        assert not torch.isnan(logvar).any(), "Encoder logvar contains NaN"
        assert not torch.isinf(logvar).any(), "Encoder logvar contains Inf"
        
        # Check logvar is not too extreme (prevents numerical issues)
        assert logvar.abs().max() < 20, f"Logvar too extreme: {logvar.abs().max()}"


# ============================================================================
# TEST DECODER
# ============================================================================

class TestMotionDecoder:
    """Test suite for MotionDecoder."""
    
    def test_decoder_initialization(self):
        """Test decoder is properly initialized with correct dimensions."""
        decoder = MotionDecoder(
            latent_dim=32,
            hidden_dim=128,
            output_dim=3,
            seq_len=250,
            num_layers=2,
            dropout=0.2
        )
        
        assert decoder.latent_dim == 32
        assert decoder.hidden_dim == 128
        assert decoder.output_dim == 3
        assert decoder.seq_len == 250
    
    def test_decoder_forward_pass(self):
        """Test decoder forward pass produces correct output shape."""
        decoder = MotionDecoder(latent_dim=16, hidden_dim=64, output_dim=3, seq_len=250)
        
        # Create dummy latent code: [batch=8, latent=16]
        z = torch.randn(8, 16)
        
        # Forward pass (no teacher forcing)
        reconstructed = decoder(z, target_seq=None, teacher_forcing_ratio=0.0)
        
        # Check output shape
        assert reconstructed.shape == (8, 250, 3), f"Expected (8, 250, 3), got {reconstructed.shape}"
    
    def test_decoder_with_teacher_forcing(self):
        """Test decoder with teacher forcing."""
        decoder = MotionDecoder(latent_dim=16, hidden_dim=64, output_dim=3, seq_len=250)
        
        z = torch.randn(4, 16)
        target_seq = torch.randn(4, 250, 3)
        
        # Forward pass with 100% teacher forcing
        reconstructed = decoder(z, target_seq=target_seq, teacher_forcing_ratio=1.0)
        
        assert reconstructed.shape == (4, 250, 3)
        assert not torch.isnan(reconstructed).any()


# ============================================================================
# TEST FULL VAE
# ============================================================================

class TestMotionVAE:
    """Test suite for complete VAE architecture."""
    
    def test_vae_initialization(self):
        """Test VAE is properly initialized."""
        vae = MotionVAE(
            input_dim=3,
            hidden_dim=128,
            latent_dim=32,
            seq_len=250,
            num_layers=2,
            dropout=0.2
        )
        
        assert vae.input_dim == 3
        assert vae.latent_dim == 32
        assert vae.seq_len == 250
        
        # Check encoder and decoder exist
        assert vae.encoder is not None
        assert vae.decoder is not None
    
    def test_vae_forward_pass(self):
        """Test VAE end-to-end forward pass."""
        vae = MotionVAE(input_dim=3, hidden_dim=64, latent_dim=16, seq_len=250)
        
        # Input sequence
        x = torch.randn(8, 250, 3)
        
        # Forward pass
        reconstructed, mu, logvar = vae(x, teacher_forcing_ratio=1.0)
        
        # Check shapes
        assert reconstructed.shape == (8, 250, 3)
        assert mu.shape == (8, 16)
        assert logvar.shape == (8, 16)
    
    def test_reparameterization_trick(self):
        """Test reparameterization produces valid samples."""
        vae = MotionVAE(input_dim=3, hidden_dim=64, latent_dim=16, seq_len=250)
        
        mu = torch.zeros(4, 16)
        logvar = torch.zeros(4, 16)  # log(1) = 0 → σ = 1
        
        # Sample latent code
        z = vae.reparameterize(mu, logvar)
        
        # Check shape
        assert z.shape == (4, 16)
        
        # Check z ~ N(0, 1) approximately
        # With 4 samples, mean and std should be close to 0 and 1
        assert z.mean().abs() < 1.0, f"Mean too far from 0: {z.mean()}"
        assert (z.std() - 1.0).abs() < 1.0, f"Std too far from 1: {z.std()}"
    
    def test_encode_decode(self):
        """Test separate encode and decode functions."""
        vae = MotionVAE(input_dim=3, hidden_dim=64, latent_dim=16, seq_len=250)
        
        # Original sequence
        x = torch.randn(4, 250, 3)
        
        # Encode
        z = vae.encode(x)
        assert z.shape == (4, 16)
        
        # Decode
        x_reconstructed = vae.decode(z)
        assert x_reconstructed.shape == (4, 250, 3)
    
    def test_sample_from_prior(self):
        """Test sampling from prior distribution."""
        vae = MotionVAE(input_dim=3, hidden_dim=64, latent_dim=16, seq_len=250)
        
        # Sample 10 sequences from prior
        samples = vae.sample(num_samples=10, device='cpu')
        
        assert samples.shape == (10, 250, 3)
        assert not torch.isnan(samples).any()
    
    def test_parameter_count(self):
        """Test VAE has expected number of parameters."""
        vae = MotionVAE(input_dim=3, hidden_dim=128, latent_dim=32, seq_len=250, num_layers=2)
        
        total_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
        
        # Expected: ~765k parameters (from demo output)
        assert 700_000 < total_params < 800_000, f"Expected ~765k params, got {total_params:,}"


# ============================================================================
# TEST LOSS FUNCTION
# ============================================================================

class TestVAELoss:
    """Test suite for VAE loss function."""
    
    def test_loss_computation(self):
        """Test VAE loss is computed correctly."""
        # Create dummy data
        reconstructed = torch.randn(8, 250, 3)
        target = torch.randn(8, 250, 3)
        mu = torch.randn(8, 32)
        logvar = torch.randn(8, 32)
        
        # Compute loss
        total_loss, recon_loss, kl_loss = vae_loss(reconstructed, target, mu, logvar, beta=1.0)
        
        # Check losses are scalars
        assert total_loss.dim() == 0, "Total loss should be scalar"
        assert recon_loss.dim() == 0, "Recon loss should be scalar"
        assert kl_loss.dim() == 0, "KL loss should be scalar"
        
        # Check losses are non-negative
        assert recon_loss >= 0, "Reconstruction loss should be non-negative"
        # KL loss can be negative (regularization)
        
        # Check total loss is sum of components
        expected_total = recon_loss + 1.0 * kl_loss
        assert torch.allclose(total_loss, expected_total, atol=1e-5)
    
    def test_beta_weighting(self):
        """Test β parameter weights KL divergence correctly."""
        reconstructed = torch.randn(4, 250, 3)
        target = torch.randn(4, 250, 3)
        mu = torch.randn(4, 32)
        logvar = torch.randn(4, 32)
        
        # Test with different β values
        loss_beta1, recon1, kl1 = vae_loss(reconstructed, target, mu, logvar, beta=1.0)
        loss_beta2, recon2, kl2 = vae_loss(reconstructed, target, mu, logvar, beta=2.0)
        
        # Reconstruction loss should be the same
        assert torch.allclose(recon1, recon2)
        
        # KL loss should be the same (before weighting)
        assert torch.allclose(kl1, kl2)
        
        # Total loss should differ by β × KL
        assert torch.allclose(loss_beta2, recon2 + 2.0 * kl2)
    
    def test_perfect_reconstruction(self):
        """Test loss when reconstruction is perfect."""
        x = torch.randn(4, 250, 3)
        mu = torch.zeros(4, 32)
        logvar = torch.zeros(4, 32)
        
        # Perfect reconstruction
        loss, recon, kl = vae_loss(x, x, mu, logvar, beta=1.0)
        
        # Reconstruction loss should be near zero
        assert recon < 1e-6, f"Expected near-zero recon loss, got {recon}"


# ============================================================================
# TEST DATA GENERATION
# ============================================================================

class TestDataGeneration:
    """Test suite for synthetic telemetry generation."""
    
    def test_aggressive_style_generation(self):
        """Test aggressive style data generation."""
        data = generate_aggressive_style(num_samples=10, seq_len=250)
        
        # Check shape
        assert data.shape == (10, 250, 3), f"Expected (10, 250, 3), got {data.shape}"
        
        # Check features are in valid ranges
        throttle = data[:, :, 0]
        brake = data[:, :, 1]
        lean = data[:, :, 2]
        
        assert throttle.min() >= 0 and throttle.max() <= 1, "Throttle out of range [0,1]"
        assert brake.min() >= 0 and brake.max() <= 1, "Brake out of range [0,1]"
        assert lean.min() >= -1 and lean.max() <= 1, "Lean angle out of range [-1,1]"
    
    def test_smooth_style_generation(self):
        """Test smooth style data generation."""
        data = generate_smooth_style(num_samples=10, seq_len=250)
        
        # Check shape
        assert data.shape == (10, 250, 3)
        
        # Check features are in valid ranges
        throttle = data[:, :, 0]
        brake = data[:, :, 1]
        lean = data[:, :, 2]
        
        assert throttle.min() >= 0 and throttle.max() <= 1
        assert brake.min() >= 0 and brake.max() <= 1
        assert lean.min() >= -1 and lean.max() <= 1
    
    def test_style_differences(self):
        """Test that aggressive and smooth styles have distinguishable characteristics."""
        aggressive = generate_aggressive_style(num_samples=50, seq_len=250)
        smooth = generate_smooth_style(num_samples=50, seq_len=250)
        
        # Aggressive should have higher max throttle on average
        aggressive_throttle_max = aggressive[:, :, 0].max(axis=1).mean()
        smooth_throttle_max = smooth[:, :, 0].max(axis=1).mean()
        
        assert aggressive_throttle_max > smooth_throttle_max, \
            "Aggressive style should have higher max throttle"
        
        # Aggressive should have higher max brake on average
        aggressive_brake_max = aggressive[:, :, 1].max(axis=1).mean()
        smooth_brake_max = smooth[:, :, 1].max(axis=1).mean()
        
        assert aggressive_brake_max > smooth_brake_max, \
            "Aggressive style should have higher max brake"
        
        # Aggressive should have more extreme lean angles
        aggressive_lean_abs_max = np.abs(aggressive[:, :, 2]).max(axis=1).mean()
        smooth_lean_abs_max = np.abs(smooth[:, :, 2]).max(axis=1).mean()
        
        assert aggressive_lean_abs_max > smooth_lean_abs_max, \
            "Aggressive style should have more extreme lean angles"


# ============================================================================
# TEST STYLE INTERPOLATION
# ============================================================================

class TestStyleInterpolation:
    """Test suite for style interpolation."""
    
    def test_interpolation_shape(self):
        """Test interpolation produces correct output shape."""
        vae = MotionVAE(input_dim=3, hidden_dim=64, latent_dim=16, seq_len=250)
        
        seq_aggressive = torch.randn(1, 250, 3)
        seq_smooth = torch.randn(1, 250, 3)
        
        # Interpolate
        interpolated = interpolate_styles(vae, seq_aggressive, seq_smooth, alpha=0.7)
        
        assert interpolated.shape == (1, 250, 3)
    
    def test_interpolation_alpha_extremes(self):
        """Test interpolation at extreme alpha values."""
        vae = MotionVAE(input_dim=3, hidden_dim=64, latent_dim=16, seq_len=250)
        vae.eval()
        
        seq_aggressive = torch.randn(1, 250, 3)
        seq_smooth = torch.randn(1, 250, 3)
        
        # alpha=1.0 should give aggressive encoding
        interp_100 = interpolate_styles(vae, seq_aggressive, seq_smooth, alpha=1.0)
        
        # alpha=0.0 should give smooth encoding
        interp_0 = interpolate_styles(vae, seq_aggressive, seq_smooth, alpha=0.0)
        
        # They should be different
        assert not torch.allclose(interp_100, interp_0, atol=0.1)
    
    def test_interpolation_mid_point(self):
        """Test interpolation at midpoint alpha=0.5."""
        vae = MotionVAE(input_dim=3, hidden_dim=64, latent_dim=16, seq_len=250)
        
        seq_aggressive = torch.randn(1, 250, 3)
        seq_smooth = torch.randn(1, 250, 3)
        
        # Interpolate at midpoint
        interpolated = interpolate_styles(vae, seq_aggressive, seq_smooth, alpha=0.5)
        
        # Check output is valid
        assert not torch.isnan(interpolated).any()
        assert interpolated.shape == (1, 250, 3)


# ============================================================================
# TEST TRAINER
# ============================================================================

class TestVAETrainer:
    """Test suite for VAE training utilities."""
    
    def test_trainer_initialization(self):
        """Test trainer is properly initialized."""
        vae = MotionVAE(input_dim=3, hidden_dim=64, latent_dim=16, seq_len=250)
        trainer = VAETrainer(vae, learning_rate=1e-3, beta=1.0, device='cpu')
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.beta == 1.0
    
    def test_training_one_epoch(self):
        """Test training for one epoch runs without errors."""
        vae = MotionVAE(input_dim=3, hidden_dim=64, latent_dim=16, seq_len=250)
        trainer = VAETrainer(vae, learning_rate=1e-3, beta=1.0, device='cpu')
        
        # Create small training dataset
        train_data = torch.randn(32, 250, 3)
        train_dataset = torch.utils.data.TensorDataset(train_data)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)
        
        # Train one epoch
        metrics = trainer.train_epoch(train_loader, teacher_forcing_ratio=1.0)
        
        # Check metrics are returned
        assert 'loss' in metrics
        assert 'recon' in metrics
        assert 'kl' in metrics
        
        # Check losses are reasonable (not NaN or Inf)
        assert not np.isnan(metrics['loss'])
        assert not np.isinf(metrics['loss'])
    
    def test_validation(self):
        """Test validation runs without errors."""
        vae = MotionVAE(input_dim=3, hidden_dim=64, latent_dim=16, seq_len=250)
        trainer = VAETrainer(vae, learning_rate=1e-3, beta=1.0, device='cpu')
        
        # Create small validation dataset
        val_data = torch.randn(16, 250, 3)
        val_dataset = torch.utils.data.TensorDataset(val_data)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8)
        
        # Validate
        metrics = trainer.validate(val_loader)
        
        # Check metrics
        assert 'loss' in metrics
        assert 'recon' in metrics
        assert 'kl' in metrics
    
    def test_fit_short_training(self):
        """Test complete training loop for a few epochs."""
        vae = MotionVAE(input_dim=3, hidden_dim=64, latent_dim=16, seq_len=250)
        trainer = VAETrainer(vae, learning_rate=1e-3, beta=1.0, device='cpu')
        
        # Create small datasets
        train_data = np.random.randn(64, 250, 3).astype(np.float32)
        val_data = np.random.randn(16, 250, 3).astype(np.float32)
        
        # Train for 3 epochs
        history = trainer.fit(
            train_data=train_data,
            val_data=val_data,
            num_epochs=3,
            batch_size=16
        )
        
        # Check history has correct length
        assert len(history['train_loss']) == 3
        assert len(history['val_loss']) == 3
        
        # Check losses decrease (roughly)
        # Note: With random data, this may not always hold, but worth checking
        # Just verify they're reasonable numbers
        assert all(not np.isnan(loss) for loss in history['train_loss'])
        assert all(not np.isnan(loss) for loss in history['val_loss'])


# ============================================================================
# INTEGRATION TEST
# ============================================================================

class TestIntegration:
    """Integration test for complete VAE pipeline."""
    
    def test_full_pipeline(self):
        """Test complete pipeline: generation → training → interpolation."""
        # Step 1: Generate data
        aggressive_data = generate_aggressive_style(num_samples=40, seq_len=250)
        smooth_data = generate_smooth_style(num_samples=40, seq_len=250)
        
        train_data = np.concatenate([aggressive_data[:30], smooth_data[:30]], axis=0)
        val_data = np.concatenate([aggressive_data[30:], smooth_data[30:]], axis=0)
        
        # Step 2: Create and train VAE
        vae = MotionVAE(input_dim=3, hidden_dim=64, latent_dim=16, seq_len=250)
        trainer = VAETrainer(vae, learning_rate=1e-3, beta=1.0, device='cpu')
        
        history = trainer.fit(
            train_data=train_data,
            val_data=val_data,
            num_epochs=5,
            batch_size=16
        )
        
        # Check training completed
        assert len(history['train_loss']) == 5
        
        # Step 3: Test interpolation
        seq_aggressive = torch.FloatTensor(aggressive_data[0:1])
        seq_smooth = torch.FloatTensor(smooth_data[0:1])
        
        interpolated = interpolate_styles(vae, seq_aggressive, seq_smooth, alpha=0.7)
        
        # Check output is valid
        assert interpolated.shape == (1, 250, 3)
        assert not torch.isnan(interpolated).any()
        
        # Check interpolated values are in valid ranges
        throttle = interpolated[0, :, 0]
        brake = interpolated[0, :, 1]
        lean = interpolated[0, :, 2]
        
        # Allow some margin due to neural network outputs
        assert throttle.min() >= -0.5 and throttle.max() <= 1.5, \
            "Throttle outside reasonable range"
        assert brake.min() >= -0.5 and brake.max() <= 1.5, \
            "Brake outside reasonable range"
        assert lean.min() >= -1.5 and lean.max() <= 1.5, \
            "Lean outside reasonable range"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
