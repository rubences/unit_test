"""Tests for racing environments."""

import pytest
import numpy as np
from moto_edge_rl.environments import RacingEnv, TrackEnv


class TestRacingEnv:
    """Tests for RacingEnv class."""
    
    def test_init_default_track(self):
        """Test environment initialization with default track."""
        env = RacingEnv()
        assert env.track_name == "default"
    
    def test_init_custom_track(self):
        """Test environment initialization with custom track."""
        env = RacingEnv(track_name="silverstone")
        assert env.track_name == "silverstone"
    
    def test_reset(self):
        """Test environment reset."""
        env = RacingEnv()
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (10,)
    
    def test_step(self):
        """Test environment step."""
        env = RacingEnv()
        env.reset()
        action = np.zeros(3)
        obs, reward, done, info = env.step(action)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)


class TestTrackEnv:
    """Tests for TrackEnv class."""
    
    def test_init(self):
        """Test track environment initialization."""
        env = TrackEnv(track_name="monza")
        assert env.track_name == "monza"
    
    def test_inherits_from_racing_env(self):
        """Test that TrackEnv inherits from RacingEnv."""
        env = TrackEnv(track_name="spa")
        assert isinstance(env, RacingEnv)
