"""
Simple test script for MotorcycleEnv without requiring gymnasium installation.
Tests basic structure and logic of the environment.
"""

import sys
import numpy as np

# Mock gymnasium for testing without installation
class MockSpaces:
    class Box:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

class MockEnv:
    metadata = {}
    def __init__(self):
        pass
    def reset(self, seed=None):
        pass

class MockGym:
    Env = MockEnv
    spaces = MockSpaces()
    
    @staticmethod
    def register(*args, **kwargs):
        pass

# Temporarily replace gymnasium with mock
sys.modules['gymnasium'] = MockGym()
sys.modules['gymnasium.spaces'] = MockSpaces()

# Now we can import our environment
from motorcycle_env import MotorcycleEnv

def test_environment():
    """Test basic environment functionality."""
    print("Testing MotorcycleEnv...")
    
    # Test 1: Environment creation
    env = MotorcycleEnv(track_name='silverstone')
    print("✓ Environment created successfully")
    
    # Test 2: Check spaces
    assert env.observation_space.shape == (8,), "Observation space shape mismatch"
    assert env.action_space.shape == (3,), "Action space shape mismatch"
    print("✓ Observation and action spaces validated")
    
    # Test 3: Reset
    # Mock np_random for testing
    env.np_random = np.random.RandomState(42)
    obs, info = env.reset()
    assert obs.shape == (8,), f"Expected obs shape (8,), got {obs.shape}"
    assert isinstance(info, dict), "Info should be a dict"
    print(f"✓ Reset successful. Initial velocity: {obs[0]:.1f} m/s")
    
    # Test 4: Step
    action = np.array([0.5, 0.5, 150.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (8,), "Observation shape mismatch after step"
    assert isinstance(reward, float), "Reward should be float"
    assert isinstance(terminated, bool), "Terminated should be bool"
    assert isinstance(truncated, bool), "Truncated should be bool"
    print(f"✓ Step successful. Reward: {reward:.3f}")
    
    # Test 5: Kamm Circle calculation
    friction = env._calculate_kamm_circle_violation(1.0, 0.5)
    assert friction > 0, "Friction usage should be positive"
    print(f"✓ Kamm Circle calculation works. Friction usage: {friction:.2f}")
    
    # Test 6: Multi-step episode
    env.reset()
    total_reward = 0
    for i in range(10):
        action = np.random.uniform(
            env.action_space.low, 
            env.action_space.high,
            size=env.action_space.shape
        )
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"✓ Multi-step episode completed. Total reward: {total_reward:.3f}")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_environment()
