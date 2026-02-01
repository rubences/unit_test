"""
Moto-Edge-RL Deployment Module

This module provides utilities for exporting trained models to edge-compatible
formats (ONNX, TensorFlow, TFLite with int8 quantization).
"""

from .export_to_edge import (
    ModelExporter,
    main
)

__all__ = [
    'ModelExporter',
    'main'
]
