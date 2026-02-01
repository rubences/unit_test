#!/usr/bin/env python3
"""
Quick start guide for the Moto-Edge-RL Training Pipeline

This module provides a high-level example of how to use each component
of the training pipeline programmatically.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def example_generate_data():
    """Example: Generate synthetic motorcycle racing data."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Generate Synthetic Data")
    print("="*70)
    
    from data.generate_synthetic_data import main as generate_data
    
    result = generate_data(
        num_laps_per_rider=10,  # Use small number for demo
        output_dir='data/processed',
        seed=42
    )
    
    print(f"\nGenerated datasets:")
    print(f"  - Pro dataset: {result['pro_dataset']}")
    print(f"  - Amateur dataset: {result['amateur_dataset']}")
    print(f"  - Total episodes: {result['total_episodes']}")


def example_train_model():
    """Example: Train hybrid offline-online RL model."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Train Hybrid RL Model")
    print("="*70)
    
    from training.train_hybrid import main as train_model
    
    result = train_model(
        pro_dataset_path='data/processed/pro_rider_dataset.hdf5',
        amateur_dataset_path='data/processed/amateur_rider_dataset.hdf5',
        output_model_path='models/moto_edge_policy.zip',
        total_timesteps=10000,  # Use small number for demo
        eval_episodes=5,
        use_bc_pretraining=True,
        seed=42
    )
    
    print(f"\nTraining results:")
    print(f"  - Model path: {result['model_path']}")
    print(f"  - Success: {result['success']}")
    if result['success']:
        print(f"  - Avg lap time: {result['eval_results']['avg_lap_time']:.2f}s")
        print(f"  - Success rate: {result['eval_results']['success_rate']*100:.1f}%")


def example_export_model():
    """Example: Export trained model to edge formats."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Export Model for Edge Deployment")
    print("="*70)
    
    from deployment.export_to_edge import main as export_model
    
    result = export_model(
        model_path='models/moto_edge_policy.zip',
        output_dir='models/edge_deployment/',
        quantize=True,
        validate=True
    )
    
    print(f"\nExported models:")
    print(f"  - ONNX: {result['onnx_path']}")
    print(f"  - TensorFlow: {result['tensorflow_path']}")
    print(f"  - TFLite: {result['tflite_path']}")
    if result['tflite_quantized']:
        print(f"  - Quantized TFLite: {result['tflite_quantized']}")


def main():
    """Run all examples."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Moto-Edge-RL Training Pipeline Quick Start Examples'
    )
    parser.add_argument(
        'example',
        nargs='?',
        choices=['data', 'train', 'export', 'all'],
        default='all',
        help='Which example to run'
    )
    
    args = parser.parse_args()
    
    try:
        if args.example in ['data', 'all']:
            example_generate_data()
        
        if args.example in ['train', 'all']:
            example_train_model()
        
        if args.example in ['export', 'all']:
            example_export_model()
        
        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
