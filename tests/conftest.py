"""Configuration for pytest."""

import pytest


@pytest.fixture
def sample_config():
    """Provide a sample configuration dictionary for tests."""
    return {
        "model": {
            "algorithm": "PPO",
            "learning_rate": 0.001
        },
        "environment": {
            "track_name": "test_track"
        },
        "training": {
            "episodes": 100
        }
    }
