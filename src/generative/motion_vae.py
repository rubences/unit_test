"""
Motion Style Transfer with Variational Autoencoder (VAE)
=========================================================

Implements a VAE-based system for learning and transferring riding styles between 
different motorcycle racing telemetry patterns. The model can learn latent representations
of 'Aggressive' (late braking, V-shape cornering) and 'Smooth' (fast through corners, 
U-shape cornering) riding styles and interpolate between them.

Architecture:
    - Encoder: Bi-LSTM → μ, log_σ²
    - Latent Space: Gaussian distribution z ~ N(μ, σ)
    - Decoder: LSTM → Reconstructed telemetry sequence
    - Loss: Reconstruction Loss (MSE) + β × KL Divergence

Author: GitHub Copilot for Competitive Motorcycle Racing
Date: 2026-01-17
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# VAE ARCHITECTURE
# ============================================================================

class MotionEncoder(nn.Module):
    """
    Encoder network that compresses telemetry sequences into latent space.
    
    Uses bidirectional LSTM to capture temporal dependencies in both directions,
    then projects to mean (μ) and log-variance (log σ²) of latent distribution.
    
    Args:
        input_dim: Number of telemetry features (e.g., 3 for throttle, brake, lean)
        hidden_dim: Hidden state dimension of LSTM
        latent_dim: Dimension of latent space z
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout probability (default: 0.2)
    """
    
    def __init__(
        self, 
        input_dim: int = 3,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Bidirectional LSTM for temporal encoding
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Project LSTM output to latent parameters
        # Bidirectional → hidden_dim * 2
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
        logger.info(f"✓ MotionEncoder initialized: {input_dim}D → LSTM({hidden_dim}×{num_layers}) → z({latent_dim}D)")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input sequence to latent distribution parameters.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
        
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log-variance of latent distribution [batch_size, latent_dim]
        """
        # LSTM encoding
        # x: [batch, seq_len, input_dim]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use final hidden state from both directions
        # h_n: [num_layers * 2, batch, hidden_dim] (2 for bidirectional)
        # Take last layer: [2, batch, hidden_dim]
        h_forward = h_n[-2, :, :]  # [batch, hidden_dim]
        h_backward = h_n[-1, :, :]  # [batch, hidden_dim]
        h_combined = torch.cat([h_forward, h_backward], dim=1)  # [batch, hidden_dim * 2]
        
        # Project to latent parameters
        mu = self.fc_mu(h_combined)  # [batch, latent_dim]
        logvar = self.fc_logvar(h_combined)  # [batch, latent_dim]
        
        return mu, logvar


class MotionDecoder(nn.Module):
    """
    Decoder network that reconstructs telemetry sequences from latent codes.
    
    Takes latent vector z and generates telemetry sequence using LSTM with 
    teacher forcing during training and autoregressive generation during inference.
    
    Args:
        latent_dim: Dimension of latent space z
        hidden_dim: Hidden state dimension of LSTM
        output_dim: Number of telemetry features to reconstruct
        seq_len: Length of output sequence (e.g., 250 for 5 sec @ 50 Hz)
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout probability (default: 0.2)
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        output_dim: int = 3,
        seq_len: int = 250,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.num_layers = num_layers
        
        # Project latent code to initial hidden state
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.fc_cell = nn.Linear(latent_dim, hidden_dim * num_layers)
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=output_dim,  # Previous output as input
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        logger.info(f"✓ MotionDecoder initialized: z({latent_dim}D) → LSTM({hidden_dim}×{num_layers}) → {output_dim}D×{seq_len}")
    
    def forward(
        self, 
        z: torch.Tensor, 
        target_seq: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> torch.Tensor:
        """
        Decode latent vector to telemetry sequence.
        
        Args:
            z: Latent code [batch_size, latent_dim]
            target_seq: Ground truth sequence for teacher forcing [batch_size, seq_len, output_dim]
            teacher_forcing_ratio: Probability of using ground truth (1.0 = always use)
        
        Returns:
            reconstructed: Generated sequence [batch_size, seq_len, output_dim]
        """
        batch_size = z.size(0)
        device = z.device
        
        # Initialize hidden and cell states from latent code
        h_0 = self.fc_hidden(z).view(self.num_layers, batch_size, self.hidden_dim)
        c_0 = self.fc_cell(z).view(self.num_layers, batch_size, self.hidden_dim)
        
        # Initialize decoder input (zeros for first timestep)
        decoder_input = torch.zeros(batch_size, 1, self.output_dim, device=device)
        
        outputs = []
        hidden = (h_0, c_0)
        
        for t in range(self.seq_len):
            # LSTM step
            lstm_out, hidden = self.lstm(decoder_input, hidden)
            
            # Project to output
            output = self.fc_out(lstm_out)  # [batch, 1, output_dim]
            outputs.append(output)
            
            # Determine next input (teacher forcing or previous output)
            if target_seq is not None and np.random.rand() < teacher_forcing_ratio:
                # Use ground truth
                decoder_input = target_seq[:, t:t+1, :]
            else:
                # Use previous prediction
                decoder_input = output
        
        # Concatenate all timesteps
        reconstructed = torch.cat(outputs, dim=1)  # [batch, seq_len, output_dim]
        
        return reconstructed


class MotionVAE(nn.Module):
    """
    Complete VAE architecture for motion style transfer.
    
    Combines encoder and decoder with reparameterization trick for 
    end-to-end training. Supports style interpolation and transfer.
    
    Args:
        input_dim: Number of input telemetry features (default: 3)
        hidden_dim: LSTM hidden dimension (default: 128)
        latent_dim: Latent space dimension (default: 32)
        seq_len: Sequence length (default: 250 for 5 sec @ 50 Hz)
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout probability (default: 0.2)
    
    Telemetry Features:
        - throttle: [0, 1] - Accelerator position
        - brake: [0, 1] - Brake pressure
        - lean_angle: [-1, 1] - Motorcycle lean angle (normalized)
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        seq_len: int = 250,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        
        # Encoder and Decoder
        self.encoder = MotionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.decoder = MotionDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            seq_len=seq_len,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"✓ MotionVAE initialized: {total_params:,} parameters")
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + ε × σ, where ε ~ N(0, 1).
        
        This allows gradients to flow through the sampling operation.
        
        Args:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log-variance [batch_size, latent_dim]
        
        Returns:
            z: Sampled latent code [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * logvar)  # σ = exp(0.5 × log(σ²))
        eps = torch.randn_like(std)  # ε ~ N(0, 1)
        z = mu + eps * std  # z = μ + ε × σ
        return z
    
    def forward(
        self, 
        x: torch.Tensor,
        teacher_forcing_ratio: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input telemetry sequence [batch_size, seq_len, input_dim]
            teacher_forcing_ratio: Probability of teacher forcing in decoder
        
        Returns:
            reconstructed: Reconstructed sequence [batch_size, seq_len, input_dim]
            mu: Latent mean [batch_size, latent_dim]
            logvar: Latent log-variance [batch_size, latent_dim]
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Sample latent code
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed = self.decoder(z, target_seq=x, teacher_forcing_ratio=teacher_forcing_ratio)
        
        return reconstructed, mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode telemetry sequence to latent code (using mean, no sampling).
        
        Args:
            x: Input sequence [batch_size, seq_len, input_dim]
        
        Returns:
            z: Latent code [batch_size, latent_dim]
        """
        mu, _ = self.encoder(x)
        return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent code to telemetry sequence.
        
        Args:
            z: Latent code [batch_size, latent_dim]
        
        Returns:
            x: Generated sequence [batch_size, seq_len, input_dim]
        """
        return self.decoder(z, target_seq=None, teacher_forcing_ratio=0.0)
    
    def sample(self, num_samples: int = 1, device: str = 'cpu') -> torch.Tensor:
        """
        Sample random telemetry sequences from prior distribution z ~ N(0, I).
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
        
        Returns:
            samples: Generated sequences [num_samples, seq_len, input_dim]
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)
        return samples


# ============================================================================
# LOSS FUNCTION
# ============================================================================

def vae_loss(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss = Reconstruction Loss + β × KL Divergence.
    
    Reconstruction Loss: MSE between reconstructed and target sequences.
    KL Divergence: D_KL(q(z|x) || p(z)) where p(z) = N(0, I).
    
    Args:
        reconstructed: Model output [batch_size, seq_len, features]
        target: Ground truth [batch_size, seq_len, features]
        mu: Latent mean [batch_size, latent_dim]
        logvar: Latent log-variance [batch_size, latent_dim]
        beta: Weight for KL divergence (β-VAE parameter)
    
    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss (MSE)
        kl_loss: KL divergence loss
    
    KL Divergence Formula:
        D_KL = -0.5 × Σ(1 + log(σ²) - μ² - σ²)
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstructed, target, reduction='mean')
    
    # KL divergence: -0.5 * sum(1 + log(var) - mu^2 - var)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / target.size(0)  # Average over batch
    
    # Total loss with β weighting
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

def generate_aggressive_style(
    num_samples: int = 100,
    seq_len: int = 250,
    sampling_rate: int = 50
) -> np.ndarray:
    """
    Generate synthetic telemetry for 'Aggressive' riding style.
    
    Characteristics:
        - Late braking: Sharp brake input at t=1.5s (corner entry)
        - V-shape cornering: Quick turn-in → apex → quick exit
        - High throttle on straights: 0.8-1.0
        - Sharp lean angle transitions: -0.8 to +0.8 (aggressive banking)
    
    Args:
        num_samples: Number of 5-second sequences to generate
        seq_len: Sequence length (default: 250 for 5 sec @ 50 Hz)
        sampling_rate: Hz (default: 50 Hz)
    
    Returns:
        data: Telemetry array [num_samples, seq_len, 3]
              Features: [throttle, brake, lean_angle]
    
    Timeline (5 seconds):
        0.0-1.5s: Straight (high throttle, no brake, neutral lean)
        1.5-2.0s: Late braking (no throttle, high brake, increasing lean)
        2.0-2.5s: Apex (V-shape: quick turn-in, max lean)
        2.5-3.5s: Exit (increasing throttle, decreasing lean)
        3.5-5.0s: Straight (high throttle, no brake, neutral lean)
    """
    data = np.zeros((num_samples, seq_len, 3))
    time = np.linspace(0, 5, seq_len)  # 5 seconds
    
    for i in range(num_samples):
        # Add variation to each sample
        brake_time = 1.5 + np.random.uniform(-0.2, 0.2)  # Late braking variation
        apex_time = 2.25 + np.random.uniform(-0.1, 0.1)
        
        for t_idx, t in enumerate(time):
            # THROTTLE: High on straights, zero during braking/cornering
            if t < brake_time:
                throttle = np.random.uniform(0.85, 1.0)  # High throttle on straight
            elif t < brake_time + 0.5:
                throttle = 0.0  # Zero during braking
            elif t < apex_time + 0.5:
                throttle = np.random.uniform(0.0, 0.3)  # Maintenance throttle
            else:
                # Aggressive throttle application on exit
                throttle = min(1.0, (t - (apex_time + 0.5)) * 0.6 + 0.3)
            
            # BRAKE: Sharp late braking pulse
            if brake_time <= t < brake_time + 0.5:
                # V-shape braking: quick on, quick off
                brake_progress = (t - brake_time) / 0.5
                if brake_progress < 0.3:
                    brake = brake_progress / 0.3 * 0.95  # Ramp up fast
                else:
                    brake = (1 - brake_progress) / 0.7 * 0.95  # Trail off
            else:
                brake = 0.0
            
            # LEAN ANGLE: V-shape cornering (sharp transitions)
            if t < brake_time:
                lean = np.random.uniform(-0.1, 0.1)  # Straight
            elif t < apex_time:
                # Sharp turn-in to max lean
                lean_progress = (t - brake_time) / (apex_time - brake_time)
                lean = -0.85 * lean_progress  # V-shape: linear increase
            elif t < apex_time + 1.0:
                # Sharp exit from max lean
                lean_progress = (t - apex_time) / 1.0
                lean = -0.85 * (1 - lean_progress)  # V-shape: linear decrease
            else:
                lean = np.random.uniform(-0.1, 0.1)  # Straight
            
            # Add noise
            throttle = np.clip(throttle + np.random.normal(0, 0.02), 0, 1)
            brake = np.clip(brake + np.random.normal(0, 0.02), 0, 1)
            lean = np.clip(lean + np.random.normal(0, 0.03), -1, 1)
            
            data[i, t_idx, :] = [throttle, brake, lean]
    
    logger.info(f"✓ Generated {num_samples} aggressive style sequences (late braking, V-shape cornering)")
    return data


def generate_smooth_style(
    num_samples: int = 100,
    seq_len: int = 250,
    sampling_rate: int = 50
) -> np.ndarray:
    """
    Generate synthetic telemetry for 'Smooth' (Fino) riding style.
    
    Characteristics:
        - Early braking: Gradual brake input at t=1.0s (corner entry)
        - U-shape cornering: Smooth turn-in → long apex → smooth exit
        - Moderate throttle: 0.6-0.8 (conservation)
        - Gradual lean angle transitions: -0.6 to +0.6 (smooth banking)
        - Longer time at apex (U-shape maintains speed through corner)
    
    Args:
        num_samples: Number of 5-second sequences to generate
        seq_len: Sequence length (default: 250 for 5 sec @ 50 Hz)
        sampling_rate: Hz (default: 50 Hz)
    
    Returns:
        data: Telemetry array [num_samples, seq_len, 3]
              Features: [throttle, brake, lean_angle]
    
    Timeline (5 seconds):
        0.0-1.0s: Straight (moderate throttle, no brake, neutral lean)
        1.0-2.0s: Early braking (gradual, no throttle, smooth lean increase)
        2.0-3.0s: Apex (U-shape: maintain speed, constant lean)
        3.0-4.0s: Exit (smooth throttle increase, gradual lean decrease)
        4.0-5.0s: Straight (moderate throttle, no brake, neutral lean)
    """
    data = np.zeros((num_samples, seq_len, 3))
    time = np.linspace(0, 5, seq_len)  # 5 seconds
    
    for i in range(num_samples):
        # Add variation to each sample
        brake_time = 1.0 + np.random.uniform(-0.1, 0.1)  # Early braking variation
        apex_start = 2.0 + np.random.uniform(-0.1, 0.1)
        apex_end = 3.0 + np.random.uniform(-0.1, 0.1)
        
        for t_idx, t in enumerate(time):
            # THROTTLE: Moderate and smooth
            if t < brake_time:
                throttle = np.random.uniform(0.65, 0.75)  # Moderate on straight
            elif t < apex_start:
                # Smooth reduction during braking
                throttle = 0.7 * (1 - (t - brake_time) / (apex_start - brake_time))
            elif t < apex_end:
                throttle = np.random.uniform(0.2, 0.4)  # Maintenance throttle at apex
            else:
                # Smooth throttle increase on exit
                throttle = min(0.75, (t - apex_end) * 0.3 + 0.3)
            
            # BRAKE: Gradual early braking
            if brake_time <= t < apex_start:
                # U-shape braking: smooth on, smooth off
                brake_progress = (t - brake_time) / (apex_start - brake_time)
                # Smoother braking curve (sinusoidal)
                brake = 0.7 * np.sin(brake_progress * np.pi)
            else:
                brake = 0.0
            
            # LEAN ANGLE: U-shape cornering (smooth transitions)
            if t < brake_time:
                lean = np.random.uniform(-0.05, 0.05)  # Straight
            elif t < apex_start:
                # Gradual turn-in
                lean_progress = (t - brake_time) / (apex_start - brake_time)
                # Smooth curve (not linear)
                lean = -0.65 * np.sin(lean_progress * np.pi / 2)
            elif t < apex_end:
                # Maintain lean at apex (U-shape)
                lean = -0.65 + np.random.uniform(-0.05, 0.05)
            elif t < apex_end + 1.0:
                # Gradual exit
                lean_progress = (t - apex_end) / 1.0
                lean = -0.65 * np.cos(lean_progress * np.pi / 2)
            else:
                lean = np.random.uniform(-0.05, 0.05)  # Straight
            
            # Add noise (less than aggressive style)
            throttle = np.clip(throttle + np.random.normal(0, 0.015), 0, 1)
            brake = np.clip(brake + np.random.normal(0, 0.015), 0, 1)
            lean = np.clip(lean + np.random.normal(0, 0.02), -1, 1)
            
            data[i, t_idx, :] = [throttle, brake, lean]
    
    logger.info(f"✓ Generated {num_samples} smooth style sequences (early braking, U-shape cornering)")
    return data


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class VAETrainer:
    """
    Training manager for MotionVAE.
    
    Handles training loop, validation, checkpointing, and logging.
    Supports separate training on 'Aggressive' and 'Smooth' datasets
    or combined training with style labels.
    
    Args:
        model: MotionVAE model to train
        learning_rate: Optimizer learning rate (default: 1e-3)
        beta: β-VAE parameter for KL weighting (default: 1.0)
        device: Training device ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        model: MotionVAE,
        learning_rate: float = 1e-3,
        beta: float = 1.0,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.beta = beta
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_recon': [],
            'train_kl': [],
            'val_loss': [],
            'val_recon': [],
            'val_kl': []
        }
        
        logger.info(f"✓ VAETrainer initialized (lr={learning_rate}, β={beta}, device={device})")
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        teacher_forcing_ratio: float = 1.0
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader with training sequences
            teacher_forcing_ratio: Probability of teacher forcing
        
        Returns:
            metrics: Dictionary with average losses
        """
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        num_batches = 0
        
        for batch in train_loader:
            # Move to device
            x = batch[0].to(self.device)  # TensorDataset returns tuple
            
            # Forward pass
            reconstructed, mu, logvar = self.model(x, teacher_forcing_ratio)
            
            # Compute loss
            loss, recon_loss, kl_loss = vae_loss(reconstructed, x, mu, logvar, self.beta)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kl = total_kl / num_batches
        
        return {
            'loss': avg_loss,
            'recon': avg_recon,
            'kl': avg_kl
        }
    
    def validate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Args:
            val_loader: DataLoader with validation sequences
        
        Returns:
            metrics: Dictionary with average losses
        """
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(self.device)  # TensorDataset returns tuple
                
                # Forward pass (no teacher forcing)
                reconstructed, mu, logvar = self.model(x, teacher_forcing_ratio=0.0)
                
                # Compute loss
                loss, recon_loss, kl_loss = vae_loss(reconstructed, x, mu, logvar, self.beta)
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kl = total_kl / num_batches
        
        return {
            'loss': avg_loss,
            'recon': avg_recon,
            'kl': avg_kl
        }
    
    def fit(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray,
        num_epochs: int = 100,
        batch_size: int = 32,
        teacher_forcing_decay: float = 0.95
    ) -> Dict[str, List[float]]:
        """
        Train VAE on dataset.
        
        Args:
            train_data: Training sequences [num_samples, seq_len, features]
            val_data: Validation sequences [num_samples, seq_len, features]
            num_epochs: Number of training epochs
            batch_size: Batch size
            teacher_forcing_decay: Decay rate for teacher forcing ratio
        
        Returns:
            history: Training history dictionary
        """
        # Create DataLoaders
        train_tensor = torch.FloatTensor(train_data)
        val_tensor = torch.FloatTensor(val_data)
        
        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        val_dataset = torch.utils.data.TensorDataset(val_tensor)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Training loop
        teacher_forcing_ratio = 1.0
        best_val_loss = float('inf')
        
        logger.info(f"Starting training: {num_epochs} epochs, batch_size={batch_size}")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, teacher_forcing_ratio)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Decay teacher forcing
            teacher_forcing_ratio *= teacher_forcing_decay
            teacher_forcing_ratio = max(teacher_forcing_ratio, 0.0)
            
            # Log progress
            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} "
                    f"(Recon: {train_metrics['recon']:.4f}, KL: {train_metrics['kl']:.4f}) | "
                    f"Val Loss: {val_metrics['loss']:.4f} "
                    f"(Recon: {val_metrics['recon']:.4f}, KL: {val_metrics['kl']:.4f}) | "
                    f"TF: {teacher_forcing_ratio:.2f}"
                )
            
            # Save history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_recon'].append(train_metrics['recon'])
            self.history['train_kl'].append(train_metrics['kl'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_recon'].append(val_metrics['recon'])
            self.history['val_kl'].append(val_metrics['kl'])
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                # Could save checkpoint here
        
        logger.info(f"✓ Training complete! Best val loss: {best_val_loss:.4f}")
        return self.history


# ============================================================================
# STYLE INTERPOLATION
# ============================================================================

def interpolate_styles(
    vae: MotionVAE,
    seq_aggressive: torch.Tensor,
    seq_smooth: torch.Tensor,
    alpha: float = 0.7,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Interpolate between aggressive and smooth riding styles in latent space.
    
    Formula: z_interpolated = α × z_aggressive + (1 - α) × z_smooth
    
    Args:
        vae: Trained MotionVAE model
        seq_aggressive: Example aggressive sequence [1, seq_len, features]
        seq_smooth: Example smooth sequence [1, seq_len, features]
        alpha: Interpolation weight (0 = smooth, 1 = aggressive)
              Example: 0.7 = 70% aggressive + 30% smooth
        device: Device for computation
    
    Returns:
        interpolated_seq: Generated sequence [1, seq_len, features]
    """
    vae.eval()
    
    with torch.no_grad():
        # Encode both styles to latent space
        seq_aggressive = seq_aggressive.to(device)
        seq_smooth = seq_smooth.to(device)
        
        z_aggressive = vae.encode(seq_aggressive)
        z_smooth = vae.encode(seq_smooth)
        
        # Linear interpolation in latent space
        z_interpolated = alpha * z_aggressive + (1 - alpha) * z_smooth
        
        # Decode interpolated latent code
        interpolated_seq = vae.decode(z_interpolated)
    
    logger.info(f"✓ Style interpolation: {alpha*100:.0f}% Aggressive + {(1-alpha)*100:.0f}% Smooth")
    return interpolated_seq


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_telemetry_comparison(
    sequences: Dict[str, np.ndarray],
    title: str = "Telemetry Style Comparison",
    save_path: Optional[str] = None
):
    """
    Plot telemetry comparison between multiple riding styles.
    
    Args:
        sequences: Dictionary {label: sequence_array}
                   sequence_array: [seq_len, 3] with [throttle, brake, lean]
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    time = np.linspace(0, 5, len(list(sequences.values())[0]))
    
    colors = {
        'Aggressive': '#e74c3c',
        'Smooth': '#3498db',
        'Interpolated': '#9b59b6',
        'Generated': '#2ecc71'
    }
    
    feature_names = ['Throttle', 'Brake', 'Lean Angle']
    
    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        
        for label, seq in sequences.items():
            color = colors.get(label, '#34495e')
            ax.plot(time, seq[:, i], label=label, linewidth=2, color=color, alpha=0.8)
        
        ax.set_ylabel(feature_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        # Style-specific annotations
        if i == 0:  # Throttle
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0.8, color='gray', linestyle='--', alpha=0.3, label='High throttle')
        elif i == 1:  # Brake
            ax.set_ylim(-0.05, 1.05)
            ax.axvspan(1.0, 2.0, alpha=0.1, color='red', label='Braking zone')
        else:  # Lean angle
            ax.set_ylim(-1.05, 1.05)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
            ax.axhspan(-0.8, -0.6, alpha=0.1, color='orange', label='Max lean')
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Plot saved to {save_path}")
    
    plt.show()


def plot_racing_line(
    sequences: Dict[str, np.ndarray],
    title: str = "Racing Line Comparison (Top View)",
    save_path: Optional[str] = None
):
    """
    Visualize racing line by integrating lean angle over time.
    
    Approximates lateral position based on lean angle and speed.
    Shows V-shape vs U-shape cornering visually.
    
    Args:
        sequences: Dictionary {label: sequence_array}
                   sequence_array: [seq_len, 3] with [throttle, brake, lean]
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = {
        'Aggressive': '#e74c3c',
        'Smooth': '#3498db',
        'Interpolated': '#9b59b6'
    }
    
    for label, seq in sequences.items():
        # Integrate lean angle to approximate lateral position
        # This is a simplified model: lateral_velocity ∝ lean_angle × speed
        throttle = seq[:, 0]
        lean = seq[:, 2]
        
        # Approximate speed (simple integration of throttle - brake)
        speed = np.cumsum(throttle - seq[:, 1]) * 0.01 + 10  # Start at 10 m/s
        speed = np.clip(speed, 0, 80)
        
        # Lateral displacement from lean angle
        lateral_velocity = lean * speed * 0.05  # Scaling factor
        lateral_pos = np.cumsum(lateral_velocity) * 0.02  # Integration
        
        # Longitudinal position
        longitudinal_pos = np.cumsum(speed) * 0.02
        
        color = colors.get(label, '#34495e')
        ax.plot(longitudinal_pos, lateral_pos, label=label, linewidth=3, color=color, alpha=0.8)
        
        # Mark start and apex
        ax.scatter(longitudinal_pos[0], lateral_pos[0], s=150, marker='o', 
                  color=color, edgecolors='white', linewidths=2, zorder=5, label=f'{label} Start')
        apex_idx = np.argmax(np.abs(lean))
        ax.scatter(longitudinal_pos[apex_idx], lateral_pos[apex_idx], s=150, marker='*', 
                  color=color, edgecolors='white', linewidths=2, zorder=5, label=f'{label} Apex')
    
    ax.set_xlabel('Longitudinal Position (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Lateral Position (m)', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11, ncol=2)
    ax.set_aspect('equal', adjustable='datalim')
    
    # Add track boundaries
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Racing line plot saved to {save_path}")
    
    plt.show()


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("MOTION STYLE TRANSFER VAE - DEMONSTRATION")
    logger.info("=" * 80)
    
    # Configuration
    LATENT_DIM = 32
    HIDDEN_DIM = 128
    SEQ_LEN = 250  # 5 seconds @ 50 Hz
    NUM_SAMPLES = 200
    NUM_EPOCHS = 20  # Reduced for demo
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Latent Dimension: {LATENT_DIM}")
    logger.info(f"  Hidden Dimension: {HIDDEN_DIM}")
    logger.info(f"  Sequence Length: {SEQ_LEN} ({SEQ_LEN/50:.1f} seconds @ 50 Hz)")
    logger.info(f"  Device: {DEVICE}")
    
    # Step 1: Generate synthetic data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Generating Synthetic Telemetry Data")
    logger.info("=" * 80)
    
    aggressive_data = generate_aggressive_style(num_samples=NUM_SAMPLES, seq_len=SEQ_LEN)
    smooth_data = generate_smooth_style(num_samples=NUM_SAMPLES, seq_len=SEQ_LEN)
    
    # Combine datasets
    train_data = np.concatenate([aggressive_data[:160], smooth_data[:160]], axis=0)
    val_data = np.concatenate([aggressive_data[160:], smooth_data[160:]], axis=0)
    
    logger.info(f"  Training samples: {len(train_data)} ({len(train_data)//2} per style)")
    logger.info(f"  Validation samples: {len(val_data)} ({len(val_data)//2} per style)")
    
    # Step 2: Create and train VAE
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Training Motion VAE")
    logger.info("=" * 80)
    
    vae = MotionVAE(
        input_dim=3,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        seq_len=SEQ_LEN,
        num_layers=2,
        dropout=0.2
    )
    
    trainer = VAETrainer(vae, learning_rate=1e-3, beta=1.0, device=DEVICE)
    history = trainer.fit(
        train_data=train_data,
        val_data=val_data,
        num_epochs=NUM_EPOCHS,
        batch_size=32
    )
    
    # Step 3: Style interpolation
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Style Interpolation (70% Aggressive + 30% Smooth)")
    logger.info("=" * 80)
    
    # Select example sequences
    seq_aggressive = torch.FloatTensor(aggressive_data[0:1])
    seq_smooth = torch.FloatTensor(smooth_data[0:1])
    
    # Interpolate
    seq_interpolated = interpolate_styles(
        vae, seq_aggressive, seq_smooth, alpha=0.7, device=DEVICE
    )
    
    # Step 4: Visualization
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Visualization")
    logger.info("=" * 80)
    
    # Convert to numpy for plotting
    sequences = {
        'Aggressive': aggressive_data[0],
        'Smooth': smooth_data[0],
        'Interpolated (70/30)': seq_interpolated.cpu().numpy()[0]
    }
    
    plot_telemetry_comparison(
        sequences,
        title="Riding Style Transfer: 70% Aggressive + 30% Smooth",
        save_path="motion_vae_telemetry.png"
    )
    
    plot_racing_line(
        sequences,
        title="Racing Line Comparison: V-shape vs U-shape Cornering",
        save_path="motion_vae_racing_line.png"
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ DEMO COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Trained VAE with {sum(p.numel() for p in vae.parameters()):,} parameters")
    logger.info(f"  Final validation loss: {history['val_loss'][-1]:.4f}")
    logger.info(f"  Plots saved: motion_vae_telemetry.png, motion_vae_racing_line.png")
