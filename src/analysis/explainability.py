"""
Explainability (XAI) analysis for motorcycle racing RL/ML models.

This module provides:
1. Attention weight visualization for Transformer/TCN models
2. SHAP (SHapley Additive exPlanations) for feature importance
3. Combined publication-ready figures

Example:
    python src/analysis/explainability.py \
        --model path/to/model.pt \
        --telemetry path/to/data.csv \
        --output outputs/explainability.pdf
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import pandas as pd

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SimpleTransformerEncoder(nn.Module):
    """Minimal Transformer encoder for telemetry analysis (example)."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=256,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_size, 1)  # Single output (decision confidence)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
        """Forward pass with attention extraction.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            (output logits, dict of attention weights per layer)
        """
        x = self.embedding(x)
        attn_maps = {}

        # Extract attention weights from each layer
        for i, layer in enumerate(self.transformer.layers):
            # Hook into self-attention to capture weights
            with torch.no_grad():
                x = layer(x)
                if hasattr(layer.self_attn, "_attn_weights"):
                    attn_maps[f"layer_{i}"] = layer.self_attn._attn_weights.cpu().numpy()

        logits = self.head(x)
        return logits, attn_maps


def load_model(model_path: Path) -> Optional[nn.Module]:
    """Load a PyTorch model; support checkpoint or full model state."""
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available; using dummy model for demo.")
        return None

    try:
        model = SimpleTransformerEncoder(input_size=6)  # 6 sensor channels
        state_dict = torch.load(model_path, map_location="cpu")
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        logger.warning(f"Failed to load model: {e}. Using dummy for demo.")
        return None


def load_telemetry(data_path: Path) -> np.ndarray:
    """Load telemetry data from CSV or NPZ.

    Expected columns: ax, ay, az, gx, gy, gz (6 channels)
    """
    if data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
        sensor_cols = ["ax", "ay", "az", "gx", "gy", "gz"]
        if not all(col in df.columns for col in sensor_cols):
            logger.warning(f"CSV missing sensor columns. Using all numeric columns.")
            sensor_cols = df.select_dtypes(np.number).columns.tolist()[:6]
        data = df[sensor_cols].values
    elif data_path.suffix == ".npz":
        data = np.load(data_path)["data"]
    else:
        raise ValueError(f"Unsupported format: {data_path.suffix}")

    return data.astype(np.float32)


def compute_attention_heatmap(
    model: nn.Module,
    telemetry: np.ndarray,
    window_size: int = 100,
) -> np.ndarray:
    """Compute attention weights over the telemetry window.

    For a sequence input, average attention across heads/layers.
    """
    if model is None or not TORCH_AVAILABLE:
        # Fallback: uniform weights
        logger.info("No model; returning uniform attention (demo).")
        return np.ones((telemetry.shape[0], 1)) / telemetry.shape[0]

    telemetry = telemetry[: window_size * (telemetry.shape[0] // window_size)]
    x = torch.from_numpy(telemetry).unsqueeze(0)  # (1, T, 6)

    with torch.no_grad():
        _, attn_maps = model(x)

    # Average attention across layers and heads
    if attn_maps:
        weights = []
        for layer_attn in attn_maps.values():
            if len(layer_attn.shape) == 4:  # (batch, heads, seq, seq)
                weights.append(layer_attn.mean(axis=(0, 1)))  # Average heads
            else:
                weights.append(layer_attn)
        avg_attn = np.mean(weights, axis=0) if weights else np.ones((telemetry.shape[0], telemetry.shape[0]))
    else:
        avg_attn = np.ones((telemetry.shape[0], telemetry.shape[0]))

    return avg_attn


def compute_shap_importance(
    telemetry: np.ndarray,
    model: Optional[nn.Module] = None,
    sensor_names: Optional[list] = None,
) -> Tuple[np.ndarray, list]:
    """Compute feature importance using SHAP.

    Args:
        telemetry: (T, 6) array of sensor readings
        model: Optional PyTorch model (if None, uses a simple heuristic)
        sensor_names: Names of sensors; defaults to ax, ay, az, gx, gy, gz

    Returns:
        (shap_values shape (6,), sensor_names)
    """
    if sensor_names is None:
        sensor_names = ["ax", "ay", "az", "gx", "gy", "gz"]

    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available. Using heuristic importance.")
        importance = np.abs(telemetry.mean(axis=0)) + np.abs(telemetry.std(axis=0))
        return importance / importance.sum(), sensor_names

    def model_fn(X):
        """Wrapper for SHAP: expects (n_samples, n_features)."""
        if model is None:
            # Simple heuristic: magnitude + variability
            return np.abs(X).mean(axis=1, keepdims=True)
        else:
            X_t = torch.from_numpy(X).unsqueeze(-1).float()
            with torch.no_grad():
                logits, _ = model(X_t)
            return logits.numpy()

    background = telemetry[: max(10, len(telemetry) // 10)]
    explainer = shap.KernelExplainer(model_fn, background)
    shap_values = explainer.shap_values(telemetry[: min(50, len(telemetry))])

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    importance = np.abs(shap_values).mean(axis=0)
    return importance / importance.sum(), sensor_names


def plot_combined_explainability(
    telemetry: np.ndarray,
    attention_heatmap: np.ndarray,
    shap_importance: np.ndarray,
    sensor_names: list,
    output_path: Path,
    title: str = "Model Explainability Analysis",
) -> None:
    """Generate a publication-ready combined figure.

    Layout:
    - Top: Raw telemetry signals
    - Middle: Attention heatmap
    - Bottom: Feature importance (bar chart)
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.5, 1.5, 1], hspace=0.4, wspace=0.3)

    # Top: Raw telemetry signals
    ax_signals = fig.add_subplot(gs[0, :])
    time = np.arange(telemetry.shape[0])
    colors = plt.cm.tab10(np.linspace(0, 1, telemetry.shape[1]))

    for i, (signal, name, color) in enumerate(zip(telemetry.T, sensor_names, colors)):
        ax_signals.plot(time, signal, label=name, color=color, linewidth=1.5, alpha=0.8)

    ax_signals.set_ylabel("Sensor Value (SI units)", fontsize=11, fontweight="bold")
    ax_signals.set_title(f"{title} – Raw Telemetry", fontsize=12, fontweight="bold")
    ax_signals.legend(loc="upper right", ncol=3, framealpha=0.9)
    ax_signals.grid(True, alpha=0.3)

    # Middle: Attention heatmap
    ax_attn = fig.add_subplot(gs[1, :])
    # Use last row of attention (final decision attention)
    attn_row = attention_heatmap[-1, :] if len(attention_heatmap.shape) == 2 else attention_heatmap
    im = ax_attn.imshow(
        attn_row[np.newaxis, :],
        aspect="auto",
        cmap="RdYlBu_r",
        extent=[0, len(attn_row), 0, 1],
    )
    ax_attn.set_ylabel("Attention\nWeight", fontsize=11, fontweight="bold")
    ax_attn.set_xlabel("Time Step", fontsize=11, fontweight="bold")
    ax_attn.set_title("Model Attention Weights Over Time", fontsize=12, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax_attn, orientation="horizontal", pad=0.1)
    cbar.set_label("Attention Magnitude", fontsize=10)

    # Bottom: Feature importance
    ax_importance = fig.add_subplot(gs[2, 0])
    bars = ax_importance.barh(sensor_names, shap_importance, color=colors, edgecolor="black", linewidth=1.2)
    ax_importance.set_xlabel("Relative Importance", fontsize=11, fontweight="bold")
    ax_importance.set_title("SHAP Feature Importance", fontsize=12, fontweight="bold")
    ax_importance.set_xlim(0, max(shap_importance) * 1.1)
    for i, (bar, val) in enumerate(zip(bars, shap_importance)):
        ax_importance.text(val + 0.01, i, f"{val:.3f}", va="center", fontsize=9)

    # Bottom-right: Summary table
    ax_summary = fig.add_subplot(gs[2, 1])
    ax_summary.axis("off")
    summary_text = f"""
Model Explainability Summary

Input Size: {telemetry.shape[0]} timesteps × {telemetry.shape[1]} sensors
Max Attention: {attention_heatmap.max():.4f}
Top Sensor: {sensor_names[np.argmax(shap_importance)]}
  Importance: {shap_importance.max():.4f}

Interpretation:
The heatmap highlights critical phases
in the maneuver. Sensors above show
relative contribution to the decision.
    """
    ax_summary.text(0.05, 0.95, summary_text, fontsize=10, verticalalignment="top", family="monospace",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle(f"{title}", fontsize=14, fontweight="bold", y=0.995)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved explainability figure to {output_path}")
    plt.close()


def generate_synthetic_telemetry(n_samples: int = 500, n_sensors: int = 6) -> np.ndarray:
    """Generate synthetic telemetry for demo when real data is unavailable."""
    t = np.arange(n_samples) / 50.0  # 50 Hz
    # Simulate a maneuver: acceleration, braking, cornering
    ax = 2.0 * np.sin(0.02 * t) + 0.5 * np.random.randn(n_samples)
    ay = 1.0 * np.sin(0.015 * t + np.pi / 4) + 0.3 * np.random.randn(n_samples)
    az = 9.81 + 0.2 * np.random.randn(n_samples)
    gx = 0.1 * np.sin(0.01 * t) + 0.05 * np.random.randn(n_samples)
    gy = 0.2 * np.sin(0.02 * t) + 0.05 * np.random.randn(n_samples)
    gz = 0.05 * np.sin(0.015 * t) + 0.02 * np.random.randn(n_samples)
    return np.column_stack([ax, ay, az, gx, gy, gz]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="XAI explainability analysis for motorcycle RL models")
    parser.add_argument("--model", type=Path, help="Path to trained model checkpoint")
    parser.add_argument("--telemetry", type=Path, help="Path to telemetry CSV/NPZ")
    parser.add_argument("--output", type=Path, default="outputs/explainability.pdf", help="Output figure path")
    args = parser.parse_args()

    # Load telemetry
    if args.telemetry and args.telemetry.exists():
        logger.info(f"Loading telemetry from {args.telemetry}")
        telemetry = load_telemetry(args.telemetry)
    else:
        logger.info("Generating synthetic telemetry for demo")
        telemetry = generate_synthetic_telemetry(n_samples=500, n_sensors=6)

    # Load model
    model = None
    if args.model and args.model.exists():
        logger.info(f"Loading model from {args.model}")
        model = load_model(args.model)

    # Compute attention heatmap
    logger.info("Computing attention heatmap...")
    attention_heatmap = compute_attention_heatmap(model, telemetry, window_size=100)

    # Compute SHAP importance
    logger.info("Computing SHAP feature importance...")
    shap_importance, sensor_names = compute_shap_importance(telemetry, model=model)

    # Generate figure
    logger.info(f"Generating publication-ready figure...")
    plot_combined_explainability(
        telemetry,
        attention_heatmap,
        shap_importance,
        sensor_names,
        args.output,
        title="Motorcycle Racing Model Explainability",
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
