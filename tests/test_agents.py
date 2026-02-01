"""Tests for RL agents."""

import pytest
from moto_edge_rl.agents import BaseAgent, PPOAgent, SACAgent


class TestBaseAgent:
    """Tests for BaseAgent class."""
    
    def test_init_default_name(self):
        """Test agent initialization with default name."""
        agent = BaseAgent()
        assert agent.name == "BaseAgent"
    
    def test_init_custom_name(self):
        """Test agent initialization with custom name."""
        agent = BaseAgent(name="CustomAgent")
        assert agent.name == "CustomAgent"
    
    def test_train_not_implemented(self):
        """Test that train raises NotImplementedError."""
        agent = BaseAgent()
        with pytest.raises(NotImplementedError):
            agent.train()
    
    def test_predict_not_implemented(self):
        """Test that predict raises NotImplementedError."""
        agent = BaseAgent()
        with pytest.raises(NotImplementedError):
            agent.predict(None)


class TestPPOAgent:
    """Tests for PPOAgent class."""
    
    def test_init_default_name(self):
        """Test PPO agent initialization with default name."""
        agent = PPOAgent()
        assert agent.name == "PPOAgent"
    
    def test_init_custom_name(self):
        """Test PPO agent initialization with custom name."""
        agent = PPOAgent(name="CustomPPO")
        assert agent.name == "CustomPPO"


class TestSACAgent:
    """Tests for SACAgent class."""
    
    def test_init_default_name(self):
        """Test SAC agent initialization with default name."""
        agent = SACAgent()
        assert agent.name == "SACAgent"
    
    def test_init_custom_name(self):
        """Test SAC agent initialization with custom name."""
        agent = SACAgent(name="CustomSAC")
        assert agent.name == "CustomSAC"
