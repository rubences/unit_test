"""
Unit Tests for Adversarial Training Components

Verifica:
1. SensorNoiseAgent attack generation
2. Curriculum progression
3. AdversarialEnvironmentWrapper integration
"""

import pytest
import numpy as np
from typing import Dict, Any

from src.agents.sensor_noise_agent import SensorNoiseAgent, AdversarialEnvironmentWrapper


class TestSensorNoiseAgent:
    """Test SensorNoiseAgent attack capabilities."""

    def test_initialization(self):
        """Test agent initialization with various parameters."""
        agent = SensorNoiseAgent(noise_level=0.15, curriculum_stage=2)
        
        assert agent.noise_level == 0.15
        assert agent.curriculum_stage == 2
        assert agent.step_count == 0
        assert len(agent.attack_modes) == 4

    def test_attack_modes_subset(self):
        """Test with subset of attack modes."""
        agent = SensorNoiseAgent(
            attack_modes=["gaussian", "cutout"]
        )
        
        assert set(agent.attack_modes) == {"gaussian", "cutout"}

    def test_inject_noise_gaussian(self):
        """Test Gaussian noise injection."""
        agent = SensorNoiseAgent(
            noise_level=0.1,
            attack_modes=["gaussian"],
        )
        
        telemetry = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        
        # Run multiple times to check randomness
        corrupted_list = []
        for _ in range(10):
            agent.rng = np.random.RandomState(np.random.randint(0, 10000))
            corrupted, metadata = agent.inject_noise(telemetry)
            corrupted_list.append(corrupted)
            
            # Check attack was applied
            assert "gaussian" in metadata["attacks_applied"]
            # Corrupted should be different from clean
            assert not np.allclose(corrupted, telemetry)

    def test_inject_noise_cutout(self):
        """Test signal cutout (sensor failure)."""
        agent = SensorNoiseAgent(
            noise_level=0.5,  # Higher probability of cutout
            attack_modes=["cutout"],
            curriculum_stage=3,  # Hard: 30% cutout probability
        )
        
        telemetry = np.ones(6, dtype=np.float32)
        
        # Run multiple times to likely trigger cutout
        cutout_occurred = False
        for _ in range(50):
            agent.rng = np.random.RandomState(np.random.randint(0, 10000))
            corrupted, metadata = agent.inject_noise(telemetry)
            
            if "cutout" in str(metadata["attacks_applied"]):
                cutout_occurred = True
                # Some values should be zero (cutout)
                assert np.any(corrupted == 0.0)
                break
        
        assert cutout_occurred, "Cutout attack should occur with high probability"

    def test_inject_noise_drift(self):
        """Test accumulated drift over time."""
        agent = SensorNoiseAgent(
            noise_level=0.1,
            attack_modes=["drift"],
        )
        
        telemetry = np.zeros(6, dtype=np.float32)
        
        drifts = []
        for _ in range(10):
            corrupted, metadata = agent.inject_noise(telemetry)
            # Drift should accumulate
            drift_magnitude = np.linalg.norm(corrupted)
            drifts.append(drift_magnitude)
        
        # Drift should generally increase
        assert drifts[-1] > drifts[0]

    def test_inject_noise_bias(self):
        """Test constant bias injection."""
        agent = SensorNoiseAgent(
            noise_level=0.2,
            attack_modes=["bias"],
        )
        
        telemetry = np.zeros(6, dtype=np.float32)
        
        # Bias should be consistent across calls (no randomness in direction)
        agent.rng = np.random.RandomState(42)
        corrupted1, _ = agent.inject_noise(telemetry.copy())
        
        agent.rng = np.random.RandomState(42)
        corrupted2, _ = agent.inject_noise(telemetry.copy())
        
        # Same seed = same bias
        assert np.allclose(corrupted1, corrupted2)

    def test_curriculum_stages(self):
        """Test curriculum progression."""
        agent = SensorNoiseAgent(noise_level=1.0)
        
        for stage in [1, 2, 3]:
            agent.set_curriculum_stage(stage)
            assert agent.curriculum_stage == stage
            params = agent.stage_params
            
            # Verify stage parameters increase with stage
            if stage == 1:
                params_1 = params.copy()
            elif stage == 2:
                params_2 = params.copy()
                assert params_2["gaussian_scale"] > params_1["gaussian_scale"]
                assert params_2["drift_rate"] > params_1["drift_rate"]
                assert params_2["cutout_prob"] > params_1["cutout_prob"]
            elif stage == 3:
                assert params["gaussian_scale"] > params_2["gaussian_scale"]
                assert params["drift_rate"] > params_2["drift_rate"]
                assert params["cutout_prob"] > params_2["cutout_prob"]

    def test_drift_reset(self):
        """Test drift accumulator reset."""
        agent = SensorNoiseAgent(
            noise_level=0.1,
            attack_modes=["drift"],
        )
        
        telemetry = np.zeros(6, dtype=np.float32)
        
        # Accumulate drift
        for _ in range(5):
            agent.inject_noise(telemetry)
        
        # Reset drift
        agent.reset_drift()
        
        assert len(agent.drift_accumulators) == 0
        assert agent.step_count == 0

    def test_attack_strength_scaling(self):
        """Test attack strength scales with noise_level and curriculum."""
        agent1 = SensorNoiseAgent(noise_level=0.1, curriculum_stage=1)
        agent2 = SensorNoiseAgent(noise_level=0.1, curriculum_stage=3)
        
        strength1 = agent1.get_attack_strength()
        strength2 = agent2.get_attack_strength()
        
        # Higher stage = stronger attack
        assert strength2 > strength1

    def test_metadata_completeness(self):
        """Test that metadata contains all expected fields."""
        agent = SensorNoiseAgent(noise_level=0.1)
        
        telemetry = np.random.randn(6).astype(np.float32)
        corrupted, metadata = agent.inject_noise(telemetry)
        
        required_fields = [
            "noise_level",
            "curriculum_stage",
            "attacks_applied",
            "perturbation_magnitude",
        ]
        
        for field in required_fields:
            assert field in metadata, f"Missing field: {field}"

    def test_status_dict(self):
        """Test get_status returns complete information."""
        agent = SensorNoiseAgent(noise_level=0.15, curriculum_stage=2)
        
        status = agent.get_status()
        
        assert "noise_level" in status
        assert "curriculum_stage" in status
        assert "attack_modes" in status
        assert "step_count" in status
        assert "stage_params" in status
        assert "attack_strength" in status


class TestAdversarialEnvironmentWrapper:
    """Test AdversarialEnvironmentWrapper integration."""

    @pytest.fixture
    def dummy_env(self):
        """Create a simple dummy environment for testing."""
        import gymnasium as gym
        from gymnasium import spaces

        class DummyEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = spaces.Box(
                    low=-1, high=1, shape=(6,), dtype=np.float32
                )
                self.action_space = spaces.Discrete(2)

            def reset(self, seed=None, options=None):
                obs = self.observation_space.sample()
                return obs, {}

            def step(self, action):
                obs = self.observation_space.sample()
                reward = 1.0
                return obs, reward, False, False, {}

        return DummyEnv()

    def test_wrapper_initialization(self, dummy_env):
        """Test wrapper initialization."""
        agent = SensorNoiseAgent(noise_level=0.1)
        wrapper = AdversarialEnvironmentWrapper(dummy_env, sensor_noise_agent=agent)
        
        assert wrapper.env == dummy_env
        assert wrapper.sensor_noise_agent == agent
        assert len(wrapper.episode_attacks) == 0

    def test_wrapper_default_agent(self, dummy_env):
        """Test wrapper creates default agent if not provided."""
        wrapper = AdversarialEnvironmentWrapper(dummy_env)
        
        assert wrapper.sensor_noise_agent is not None
        assert isinstance(wrapper.sensor_noise_agent, SensorNoiseAgent)

    def test_wrapper_reset(self, dummy_env):
        """Test wrapper reset clears attack tracking."""
        agent = SensorNoiseAgent()
        wrapper = AdversarialEnvironmentWrapper(dummy_env, sensor_noise_agent=agent)
        
        # Add dummy attacks
        wrapper.episode_attacks.append({"test": "attack"})
        assert len(wrapper.episode_attacks) > 0
        
        # Reset
        obs, info = wrapper.reset()
        
        # Attacks cleared
        assert len(wrapper.episode_attacks) == 0
        # Drift reset
        assert len(agent.drift_accumulators) == 0

    def test_wrapper_step_adds_adversarial_info(self, dummy_env):
        """Test wrapper.step() adds adversarial info to step output."""
        wrapper = AdversarialEnvironmentWrapper(dummy_env)
        wrapper.reset()
        
        obs, reward, terminated, truncated, info = wrapper.step(action=0)
        
        assert "adversarial" in info
        assert "perturbation_magnitude" in info["adversarial"]

    def test_wrapper_noise_level_update(self, dummy_env):
        """Test updating noise level through wrapper."""
        wrapper = AdversarialEnvironmentWrapper(dummy_env)
        
        wrapper.set_noise_level(0.25)
        assert wrapper.sensor_noise_agent.noise_level == 0.25

    def test_wrapper_curriculum_update(self, dummy_env):
        """Test updating curriculum stage through wrapper."""
        wrapper = AdversarialEnvironmentWrapper(dummy_env)
        
        wrapper.set_curriculum_stage(3)
        assert wrapper.sensor_noise_agent.curriculum_stage == 3

    def test_wrapper_episode_stats(self, dummy_env):
        """Test episode statistics computation."""
        wrapper = AdversarialEnvironmentWrapper(dummy_env)
        wrapper.reset()
        
        # Run a few steps
        for _ in range(5):
            wrapper.step(action=0)
        
        stats = wrapper.get_episode_stats()
        
        assert "total_attacks" in stats
        assert "avg_perturbation" in stats
        assert "max_perturbation" in stats
        assert "min_perturbation" in stats
        assert stats["total_attacks"] == 5


class TestCurriculumLearning:
    """Test curriculum learning mechanics."""

    def test_curriculum_schedule(self):
        """Test that curriculum stages have increasing difficulty."""
        stages_params = {}
        
        for stage in [1, 2, 3]:
            agent = SensorNoiseAgent(curriculum_stage=stage)
            stages_params[stage] = agent.stage_params
        
        # Verify monotonic increase
        for key in ["gaussian_scale", "drift_rate", "cutout_prob", "bias_magnitude"]:
            values = [stages_params[s][key] for s in [1, 2, 3]]
            assert values[0] < values[1] < values[2], f"{key} not increasing"

    def test_curriculum_callback_simulation(self):
        """Simulate curriculum advancement during training."""
        agent = SensorNoiseAgent(noise_level=0.1)
        
        # Simulate training progress
        initial_stage = agent.curriculum_stage
        
        # After 10k timesteps, should advance to stage 2
        agent.set_curriculum_stage(2)
        assert agent.curriculum_stage == 2
        
        # After 20k timesteps, should advance to stage 3
        agent.set_curriculum_stage(3)
        assert agent.curriculum_stage == 3


class TestRobustnessMetrics:
    """Test robustness evaluation metrics."""

    def test_perturbation_magnitude(self):
        """Test perturbation magnitude is computed correctly."""
        agent = SensorNoiseAgent(noise_level=0.5, attack_modes=["gaussian"])
        
        telemetry = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float32)
        corrupted, metadata = agent.inject_noise(telemetry)
        
        # Perturbation magnitude should be norm of difference
        perturbation = corrupted - telemetry
        expected_magnitude = np.linalg.norm(perturbation)
        
        assert np.isclose(
            metadata["perturbation_magnitude"],
            expected_magnitude,
            rtol=1e-5
        )

    def test_attack_tracking(self):
        """Test that attacks are properly tracked in metadata."""
        agent = SensorNoiseAgent(
            noise_level=0.3,
            attack_modes=["gaussian", "drift", "cutout", "bias"],
        )
        
        telemetry = np.random.randn(6).astype(np.float32)
        
        for _ in range(100):
            corrupted, metadata = agent.inject_noise(telemetry)
            attacks = metadata["attacks_applied"]
            
            # Should have at least some attacks
            assert len(attacks) > 0
            # Attacks should be from allowed modes
            for attack in attacks:
                assert any(
                    mode in attack for mode in ["gaussian", "drift", "cutout", "bias"]
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
