"""
Multi-Agent Motorcycle Racing Environment using PettingZoo

5 motorcycles competing simultaneously with:
- V2V safety system (GNN-based collision prediction)
- Dynamic proximity alerts
- Haptic feedback integration
- Competitive racing dynamics
"""

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from typing import Dict, List, Tuple, Optional
import logging

try:
    from src.safety.gnn_v2v import V2VSafetySystem, GNNPolicy, generate_haptic_pattern
    V2V_AVAILABLE = True
except ImportError:
    V2V_AVAILABLE = False
    logging.warning("V2V safety system not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiMotoRacingEnv(ParallelEnv):
    """
    Multi-agent motorcycle racing environment with V2V safety.
    
    Agents: 5 motorcycles (moto_0, moto_1, ..., moto_4)
    
    Observation Space (per agent): Box(8,)
        [0] own_pos_x: X coordinate (meters)
        [1] own_pos_y: Y coordinate (meters)
        [2] own_vel_x: Velocity X (m/s)
        [3] own_vel_y: Velocity Y (m/s)
        [4] own_heading: Heading angle (radians)
        [5] track_progress: Progress along track [0, 1]
        [6] collision_risk: GNN-predicted risk [0, 1]
        [7] proximity_alert: 1 if alert active, 0 otherwise
    
    Action Space (per agent): Box(4,)
        [0] throttle: [0, 1]
        [1] brake: [0, 1]
        [2] steering: [-1, 1] (left/right)
        [3] manual_haptic: [0, 1] (overridden by safety system)
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'name': 'multi_moto_racing_v0',
    }
    
    def __init__(
        self,
        num_agents: int = 5,
        track_length: float = 1000.0,
        track_width: float = 20.0,
        proximity_threshold: float = 10.0,
        enable_v2v: bool = True,
        max_steps: int = 1000,
    ):
        """
        Initialize multi-agent racing environment.
        
        Args:
            num_agents: Number of motorcycles (default: 5)
            track_length: Track length in meters
            track_width: Track width in meters
            proximity_threshold: Distance threshold for V2V alerts (meters)
            enable_v2v: Enable V2V safety system
            max_steps: Maximum episode steps
        """
        super().__init__()
        
        self._num_agents = num_agents
        self.track_length = track_length
        self.track_width = track_width
        self.proximity_threshold = proximity_threshold
        self.enable_v2v = enable_v2v and V2V_AVAILABLE
        self.max_steps = max_steps
        
        # Agent IDs
        self.possible_agents = [f"moto_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents.copy()
        
        # Observation and action spaces
        self._obs_space = spaces.Box(
            low=np.array([0.0, -track_width/2, 0.0, -50.0, -np.pi, 0.0, 0.0, 0.0]),
            high=np.array([track_length, track_width/2, 80.0, 50.0, np.pi, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        self._action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Initialize V2V safety system
        if self.enable_v2v:
            logger.info("[MultiMotoRacingEnv] Initializing V2V safety system...")
            gnn_model = GNNPolicy(node_features=4, hidden_dim=64, output_dim=1)
            self.safety_system = V2VSafetySystem(
                gnn_model=gnn_model,
                proximity_threshold=proximity_threshold,
                collision_threshold=0.7,
                haptic_alert_intensity=0.9,
            )
            logger.info("  ✓ V2V safety system initialized")
        else:
            self.safety_system = None
            logger.info("[MultiMotoRacingEnv] V2V safety system disabled")
        
        # State variables
        self.step_count = 0
        self.positions = None
        self.velocities = None
        self.headings = None
        self.track_progress = None
        self.collision_risks = None
        self.proximity_alerts = None
        
        logger.info(f"✓ MultiMotoRacingEnv initialized ({num_agents} agents)")
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset environment to initial state.
        
        Returns:
            observations: Dict of observations per agent
            infos: Dict of info dicts per agent
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset agents
        self.agents = self.possible_agents.copy()
        self.step_count = 0
        
        # Initialize positions (staggered start)
        self.positions = np.array([
            [i * 5.0, (i - 2) * 2.0]  # Staggered X, centered Y
            for i in range(self._num_agents)
        ])
        
        # Initialize velocities (all start at similar speed)
        self.velocities = np.array([
            [15.0 + np.random.randn() * 0.5, 0.0]
            for _ in range(self._num_agents)
        ])
        
        # Initialize headings (all facing forward)
        self.headings = np.zeros(self._num_agents)
        
        # Initialize track progress
        self.track_progress = np.zeros(self._num_agents)
        
        # Initialize collision risks and alerts
        self.collision_risks = np.zeros(self._num_agents)
        self.proximity_alerts = np.zeros(self._num_agents, dtype=bool)
        
        # Get observations
        observations = self._get_observations()
        infos = self._get_infos()
        
        return observations, infos
    
    def step(self, actions: Dict[str, np.ndarray]):
        """
        Execute one environment step.
        
        Args:
            actions: Dictionary of actions per agent
                {agent_id: [throttle, brake, steering, manual_haptic]}
        
        Returns:
            observations: Dict of observations per agent
            rewards: Dict of rewards per agent
            terminations: Dict of done flags per agent
            truncations: Dict of truncation flags per agent
            infos: Dict of info dicts per agent
        """
        self.step_count += 1
        
        # Update V2V safety system
        if self.enable_v2v and self.safety_system is not None:
            # Predict collision risks
            risks = self.safety_system.predict_collision_risk(
                self.positions, self.velocities
            )
            
            # Get proximity alerts
            alerts = self.safety_system.get_proximity_alerts(
                self.positions, self.velocities
            )
            
            # Update state
            for i, agent_id in enumerate(self.agents):
                self.collision_risks[i] = risks[i]
                self.proximity_alerts[i] = alerts[i]['alert_active']
        
        # Apply actions and update dynamics
        for i, agent_id in enumerate(self.agents):
            action = actions[agent_id]
            self._update_agent_dynamics(i, action)
        
        # Compute rewards
        rewards = self._compute_rewards()
        
        # Check terminations
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: self.step_count >= self.max_steps for agent in self.agents}
        
        # Get observations and infos
        observations = self._get_observations()
        infos = self._get_infos()
        
        return observations, rewards, terminations, truncations, infos
    
    def _update_agent_dynamics(self, agent_idx: int, action: np.ndarray):
        """
        Update single agent's position and velocity.
        
        Args:
            agent_idx: Agent index
            action: Action array [throttle, brake, steering, manual_haptic]
        """
        throttle, brake, steering, _ = action
        
        # Physics parameters
        dt = 0.1  # Time step (seconds)
        max_acceleration = 5.0  # m/s^2
        max_brake = 8.0  # m/s^2
        max_steering_rate = 0.3  # rad/s
        drag_coeff = 0.02
        
        # Current state
        pos = self.positions[agent_idx]
        vel = self.velocities[agent_idx]
        heading = self.headings[agent_idx]
        
        # Acceleration
        accel_forward = throttle * max_acceleration - brake * max_brake
        
        # Drag force
        speed = np.linalg.norm(vel)
        drag = -drag_coeff * speed * vel
        
        # Update velocity
        vel_mag = np.linalg.norm(vel)
        if vel_mag > 0:
            vel_dir = vel / vel_mag
        else:
            vel_dir = np.array([1.0, 0.0])
        
        vel_new = vel + (accel_forward * vel_dir + drag) * dt
        
        # Constrain speed
        speed_new = np.linalg.norm(vel_new)
        if speed_new > 80.0:
            vel_new = vel_new / speed_new * 80.0
        elif speed_new < 0.0:
            vel_new = np.array([0.0, 0.0])
        
        # Update heading
        heading_new = heading + steering * max_steering_rate * dt
        heading_new = np.arctan2(np.sin(heading_new), np.cos(heading_new))  # Wrap to [-π, π]
        
        # Update position
        pos_new = pos + vel_new * dt
        
        # Track boundaries (simple wrapping)
        if pos_new[1] < -self.track_width / 2:
            pos_new[1] = -self.track_width / 2
            vel_new[1] = 0.0
        elif pos_new[1] > self.track_width / 2:
            pos_new[1] = self.track_width / 2
            vel_new[1] = 0.0
        
        # Update track progress
        progress = pos_new[0] / self.track_length
        progress = np.clip(progress, 0.0, 1.0)
        
        # Store updated state
        self.positions[agent_idx] = pos_new
        self.velocities[agent_idx] = vel_new
        self.headings[agent_idx] = heading_new
        self.track_progress[agent_idx] = progress
    
    def _compute_rewards(self) -> Dict[str, float]:
        """
        Compute rewards for all agents.
        
        Returns:
            rewards: Dictionary of rewards per agent
        """
        rewards = {}
        
        for i, agent_id in enumerate(self.agents):
            # Base reward: progress + speed
            progress_reward = self.track_progress[i] * 0.1
            speed = np.linalg.norm(self.velocities[i])
            speed_reward = speed / 80.0 * 0.05
            
            base_reward = progress_reward + speed_reward
            
            # Apply V2V safety penalty
            if self.enable_v2v and self.safety_system is not None:
                collision_risk = self.collision_risks[i]
                modified_reward = self.safety_system.compute_safety_reward(
                    agent_id=i,
                    base_reward=base_reward,
                    collision_risk=collision_risk,
                    penalty_weight=0.5,
                )
            else:
                modified_reward = base_reward
            
            # Proximity alert penalty (encourages safety)
            if self.proximity_alerts[i]:
                modified_reward -= 0.1
            
            rewards[agent_id] = modified_reward
        
        return rewards
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """
        Get observations for all agents.
        
        Returns:
            observations: Dictionary of observations per agent
        """
        observations = {}
        
        for i, agent_id in enumerate(self.agents):
            obs = np.array([
                self.positions[i, 0],  # pos_x
                self.positions[i, 1],  # pos_y
                self.velocities[i, 0],  # vel_x
                self.velocities[i, 1],  # vel_y
                self.headings[i],  # heading
                self.track_progress[i],  # progress
                self.collision_risks[i],  # collision risk
                float(self.proximity_alerts[i]),  # alert flag
            ], dtype=np.float32)
            
            observations[agent_id] = obs
        
        return observations
    
    def _get_infos(self) -> Dict[str, Dict]:
        """
        Get info dictionaries for all agents.
        
        Returns:
            infos: Dictionary of info dicts per agent
        """
        infos = {}
        
        # Get V2V alerts if available
        if self.enable_v2v and self.safety_system is not None:
            alerts = self.safety_system.get_proximity_alerts(
                self.positions, self.velocities
            )
        else:
            alerts = {i: {'haptic_pattern': 'none', 'haptic_intensity': 0.0} 
                     for i in range(self._num_agents)}
        
        for i, agent_id in enumerate(self.agents):
            infos[agent_id] = {
                'position': self.positions[i].tolist(),
                'velocity': self.velocities[i].tolist(),
                'heading': float(self.headings[i]),
                'track_progress': float(self.track_progress[i]),
                'collision_risk': float(self.collision_risks[i]),
                'proximity_alert': bool(self.proximity_alerts[i]),
                'haptic_pattern': alerts[i]['haptic_pattern'],
                'haptic_intensity': alerts[i]['haptic_intensity'],
                'step': self.step_count,
            }
        
        return infos
    
    def observation_space(self, agent: str):
        """Get observation space for agent."""
        return self._obs_space
    
    def action_space(self, agent: str):
        """Get action space for agent."""
        return self._action_space
    
    def render(self):
        """Render environment (placeholder)."""
        if self.step_count % 100 == 0:
            logger.info(f"\nStep {self.step_count}:")
            for i, agent_id in enumerate(self.agents):
                logger.info(f"  {agent_id}: pos=({self.positions[i,0]:.1f}, {self.positions[i,1]:.1f}), "
                          f"risk={self.collision_risks[i]:.2f}, alert={self.proximity_alerts[i]}")


if __name__ == '__main__':
    logger.info("="*70)
    logger.info("Multi-Agent Motorcycle Racing Environment Demo")
    logger.info("="*70)
    
    # Create environment
    logger.info("\n[1] Creating environment with 5 motorcycles...")
    env = MultiMotoRacingEnv(
        num_agents=5,
        track_length=1000.0,
        enable_v2v=True,
        max_steps=200,
    )
    logger.info(f"  ✓ Environment created")
    logger.info(f"    Agents: {env.possible_agents}")
    logger.info(f"    Observation space: {env.observation_space('moto_0')}")
    logger.info(f"    Action space: {env.action_space('moto_0')}")
    
    # Reset environment
    logger.info("\n[2] Resetting environment...")
    observations, infos = env.reset(seed=42)
    logger.info(f"  ✓ Environment reset")
    logger.info(f"    Initial positions:")
    for agent_id, info in infos.items():
        logger.info(f"      {agent_id}: {info['position']}")
    
    # Run episode
    logger.info("\n[3] Running 50 steps...")
    total_rewards = {agent: 0.0 for agent in env.agents}
    alert_counts = {agent: 0 for agent in env.agents}
    
    for step in range(50):
        # Random actions
        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }
        
        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Accumulate statistics
        for agent in env.agents:
            total_rewards[agent] += rewards[agent]
            if infos[agent]['proximity_alert']:
                alert_counts[agent] += 1
        
        # Render every 10 steps
        if step % 10 == 0:
            env.render()
    
    # Final statistics
    logger.info("\n[4] Episode Statistics:")
    logger.info("  Total Rewards:")
    for agent, reward in total_rewards.items():
        logger.info(f"    {agent}: {reward:.2f}")
    
    logger.info("\n  Proximity Alerts Triggered:")
    for agent, count in alert_counts.items():
        logger.info(f"    {agent}: {count} times")
    
    logger.info("\n  Final Positions:")
    for agent, info in infos.items():
        logger.info(f"    {agent}: ({info['position'][0]:.1f}, {info['position'][1]:.1f}) m, "
                  f"progress={info['track_progress']*100:.1f}%")
    
    logger.info("\n" + "="*70)
    logger.info("✓ Multi-Agent Environment Demo Complete")
    logger.info("="*70)
