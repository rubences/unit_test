#!/usr/bin/env python3
"""Script to download sample racing data."""

import argparse
from pathlib import Path


def download_sample_data(output_dir: str = "data/raw") -> None:
    """Download sample racing data for testing.
    
    Args:
        output_dir: Directory to save downloaded data.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading sample data to {output_dir}...")
    print("Note: This is a placeholder. Implement actual download logic.")
    
    # Placeholder for actual download implementation
    # In a real scenario, this would download from a dataset repository
    
    print("Sample data download complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download sample racing data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for downloaded data"
    )
    
    args = parser.parse_args()
    download_sample_data(args.output_dir)
