"""Reinforcement learning agents for motorcycle racing.

This module contains implementations of various RL algorithms optimized for
racing scenarios, including PPO, SAC, and TD3.
"""

from typing import Optional

__all__ = ["BaseAgent", "PPOAgent", "SACAgent"]


class BaseAgent:
    """Base class for all RL agents."""
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the base agent.
        
        Args:
            name: Optional name for the agent.
        """
        self.name = name or "BaseAgent"
    
    def train(self) -> None:
        """Train the agent."""
        raise NotImplementedError
    
    def predict(self, observation):
        """Make a prediction given an observation.
        
        Args:
            observation: The current state observation.
            
        Returns:
            The predicted action.
        """
        raise NotImplementedError


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization agent for racing."""
    
    def __init__(self, name: Optional[str] = None):
        """Initialize PPO agent.
        
        Args:
            name: Optional name for the agent.
        """
        super().__init__(name or "PPOAgent")


class SACAgent(BaseAgent):
    """Soft Actor-Critic agent for racing."""
    
    def __init__(self, name: Optional[str] = None):
        """Initialize SAC agent.
        
        Args:
            name: Optional name for the agent.
        """
        super().__init__(name or "SACAgent")
