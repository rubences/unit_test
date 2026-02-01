"""
V2V Safety System using Graph Neural Networks (GNN)

Implements vehicle-to-vehicle communication and collision prediction using:
- PyTorch Geometric for GNN architecture
- Dynamic graph construction based on proximity (<10m)
- Collision risk prediction for t+1
- Integration with RL reward system and haptic feedback

Architecture:
    Nodes: Motorcycles with features [pos_x, pos_y, vel_x, vel_y]
    Edges: Dynamic connections when distance < 10 meters
    Output: Collision probability [0, 1] for each motorcycle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GNNPolicy(nn.Module):
    """
    Graph Neural Network for collision risk prediction in multi-agent racing.
    
    Architecture:
        - 2 GCN layers for message passing between nearby motorcycles
        - MLP for collision probability prediction
        - Attention mechanism for weighted neighbor aggregation
    
    Node Features (4D):
        - pos_x: X coordinate (meters)
        - pos_y: Y coordinate (meters)
        - vel_x: Velocity X component (m/s)
        - vel_y: Velocity Y component (m/s)
    
    Output:
        - collision_prob: Probability of collision in next timestep [0, 1]
    """
    
    def __init__(
        self,
        node_features: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize GNN collision predictor.
        
        Args:
            node_features: Input feature dimension per node (default: 4)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (1 for collision probability)
            num_layers: Number of GCN layers
            dropout: Dropout rate for regularization
        """
        super(GNNPolicy, self).__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Graph Convolutional Layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Batch normalization for stable training
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # MLP for collision prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid(),  # Output in [0, 1]
        )
        
        logger.info(f"✓ GNNPolicy initialized: {node_features}→{hidden_dim}→{output_dim}")
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through GNN.
        
        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, node_features]
                - edge_index: Edge connectivity [2, num_edges]
                - batch: Batch assignment [num_nodes]
        
        Returns:
            collision_probs: Collision probability per node [num_nodes, 1]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Message passing through GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=0.1, training=self.training)
        
        # Predict collision probability
        collision_probs = self.mlp(x)
        
        return collision_probs


class V2VGraph:
    """
    Dynamic graph constructor for vehicle-to-vehicle communication.
    
    Creates edges between motorcycles within proximity threshold.
    """
    
    def __init__(self, proximity_threshold: float = 10.0):
        """
        Initialize V2V graph constructor.
        
        Args:
            proximity_threshold: Maximum distance for edge creation (meters)
        """
        self.proximity_threshold = proximity_threshold
        logger.info(f"✓ V2VGraph initialized (proximity={proximity_threshold}m)")
    
    def construct_graph(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        agent_ids: Optional[List[int]] = None,
    ) -> Data:
        """
        Construct dynamic graph from motorcycle states.
        
        Args:
            positions: Array of positions [num_agents, 2] (x, y)
            velocities: Array of velocities [num_agents, 2] (vx, vy)
            agent_ids: Optional list of agent IDs (default: 0, 1, 2, ...)
        
        Returns:
            data: PyTorch Geometric Data object
        """
        num_agents = len(positions)
        
        if agent_ids is None:
            agent_ids = list(range(num_agents))
        
        # Node features: [pos_x, pos_y, vel_x, vel_y]
        node_features = np.hstack([positions, velocities])  # [num_agents, 4]
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Compute pairwise distances
        distances = self._compute_distances(positions)
        
        # Create edges for motorcycles within proximity threshold
        edge_index = []
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j and distances[i, j] < self.proximity_threshold:
                    edge_index.append([i, j])
        
        if len(edge_index) == 0:
            # No edges - isolated nodes
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            num_nodes=num_agents,
        )
        
        return data
    
    def _compute_distances(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Euclidean distances.
        
        Args:
            positions: Array of positions [num_agents, 2]
        
        Returns:
            distances: Distance matrix [num_agents, num_agents]
        """
        num_agents = len(positions)
        distances = np.zeros((num_agents, num_agents))
        
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    distances[i, j] = np.linalg.norm(positions[i] - positions[j])
        
        return distances


class V2VSafetySystem:
    """
    Complete V2V safety system integrating GNN with RL and haptic feedback.
    
    Features:
        - Real-time collision risk assessment
        - Proximity alert triggering
        - RL reward modulation
        - Haptic feedback pattern generation
    """
    
    def __init__(
        self,
        gnn_model: GNNPolicy,
        proximity_threshold: float = 10.0,
        collision_threshold: float = 0.7,
        haptic_alert_intensity: float = 0.9,
        device: str = 'cpu',
    ):
        """
        Initialize V2V safety system.
        
        Args:
            gnn_model: Trained GNN collision predictor
            proximity_threshold: Distance threshold for graph edges (meters)
            collision_threshold: Probability threshold for collision alert
            haptic_alert_intensity: Haptic intensity for proximity alert [0, 1]
            device: Computation device ('cpu' or 'cuda')
        """
        self.gnn_model = gnn_model.to(device)
        self.gnn_model.eval()
        
        self.graph_constructor = V2VGraph(proximity_threshold)
        self.collision_threshold = collision_threshold
        self.haptic_alert_intensity = haptic_alert_intensity
        self.device = device
        
        logger.info(f"✓ V2VSafetySystem initialized (collision_threshold={collision_threshold})")
    
    def predict_collision_risk(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> Dict[int, float]:
        """
        Predict collision risk for each motorcycle.
        
        Args:
            positions: Array of positions [num_agents, 2]
            velocities: Array of velocities [num_agents, 2]
        
        Returns:
            risks: Dictionary {agent_id: collision_probability}
        """
        # Construct dynamic graph
        graph = self.graph_constructor.construct_graph(positions, velocities)
        graph = graph.to(self.device)
        
        # Predict collision probabilities
        with torch.no_grad():
            collision_probs = self.gnn_model(graph)
        
        # Convert to dictionary
        risks = {
            i: float(collision_probs[i].cpu().item())
            for i in range(len(positions))
        }
        
        return risks
    
    def get_proximity_alerts(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> Dict[int, Dict[str, any]]:
        """
        Generate proximity alerts for motorcycles at risk.
        
        Args:
            positions: Array of positions [num_agents, 2]
            velocities: Array of velocities [num_agents, 2]
        
        Returns:
            alerts: Dictionary with alert info per agent
                {agent_id: {
                    'risk': float,
                    'alert_active': bool,
                    'haptic_intensity': float,
                    'haptic_pattern': str
                }}
        """
        # Predict collision risks
        risks = self.predict_collision_risk(positions, velocities)
        
        # Generate alerts
        alerts = {}
        for agent_id, risk in risks.items():
            alert_active = risk >= self.collision_threshold
            
            alerts[agent_id] = {
                'risk': risk,
                'alert_active': alert_active,
                'haptic_intensity': self.haptic_alert_intensity if alert_active else 0.0,
                'haptic_pattern': 'rapid_pulse' if alert_active else 'none',
                'risk_level': self._classify_risk(risk),
            }
        
        return alerts
    
    def compute_safety_reward(
        self,
        agent_id: int,
        base_reward: float,
        collision_risk: float,
        penalty_weight: float = 0.5,
    ) -> float:
        """
        Modify RL reward based on collision risk.
        
        Args:
            agent_id: Agent ID
            base_reward: Original reward from environment
            collision_risk: Predicted collision probability [0, 1]
            penalty_weight: Weight for safety penalty
        
        Returns:
            modified_reward: Reward with safety penalty
        """
        # Safety penalty: higher risk → larger penalty
        safety_penalty = -penalty_weight * collision_risk
        
        # Modified reward
        modified_reward = base_reward + safety_penalty
        
        return modified_reward
    
    def _classify_risk(self, risk: float) -> str:
        """
        Classify collision risk level.
        
        Args:
            risk: Collision probability [0, 1]
        
        Returns:
            level: 'low', 'medium', 'high', or 'critical'
        """
        if risk < 0.3:
            return 'low'
        elif risk < 0.5:
            return 'medium'
        elif risk < 0.7:
            return 'high'
        else:
            return 'critical'


def generate_haptic_pattern(pattern_type: str, duration: float = 1.0) -> Dict[str, any]:
    """
    Generate haptic feedback pattern for proximity alerts.
    
    Args:
        pattern_type: 'rapid_pulse', 'slow_pulse', 'continuous', or 'none'
        duration: Pattern duration in seconds
    
    Returns:
        pattern: Dictionary with pattern specification
    """
    patterns = {
        'rapid_pulse': {
            'frequency': 10.0,  # Hz
            'duty_cycle': 0.5,
            'amplitude': 0.9,
            'duration': duration,
            'description': 'Fast pulsing for imminent collision'
        },
        'slow_pulse': {
            'frequency': 3.0,  # Hz
            'duty_cycle': 0.5,
            'amplitude': 0.6,
            'duration': duration,
            'description': 'Slow pulsing for moderate risk'
        },
        'continuous': {
            'frequency': 0.0,  # No pulsing
            'duty_cycle': 1.0,
            'amplitude': 0.8,
            'duration': duration,
            'description': 'Continuous vibration for high risk'
        },
        'none': {
            'frequency': 0.0,
            'duty_cycle': 0.0,
            'amplitude': 0.0,
            'duration': 0.0,
            'description': 'No haptic feedback'
        }
    }
    
    return patterns.get(pattern_type, patterns['none'])


if __name__ == '__main__':
    logger.info("="*70)
    logger.info("V2V Safety System - GNN Demo")
    logger.info("="*70)
    
    # Demo: 5 motorcycles racing
    num_agents = 5
    
    # Initialize GNN model
    logger.info("\n[1] Initializing GNN model...")
    gnn_model = GNNPolicy(
        node_features=4,
        hidden_dim=64,
        output_dim=1,
        num_layers=2,
    )
    logger.info(f"   Model parameters: {sum(p.numel() for p in gnn_model.parameters())}")
    
    # Initialize V2V safety system
    logger.info("\n[2] Initializing V2V safety system...")
    safety_system = V2VSafetySystem(
        gnn_model=gnn_model,
        proximity_threshold=10.0,
        collision_threshold=0.7,
        haptic_alert_intensity=0.9,
    )
    
    # Simulate motorcycle positions and velocities
    logger.info("\n[3] Simulating 5 motorcycles...")
    np.random.seed(42)
    positions = np.array([
        [0.0, 0.0],    # Moto 0
        [5.0, 2.0],    # Moto 1 - close to 0
        [20.0, 5.0],   # Moto 2 - far from others
        [4.0, -1.0],   # Moto 3 - close to 0 and 1
        [25.0, 10.0],  # Moto 4 - far from others
    ])
    
    velocities = np.array([
        [15.0, 0.5],   # Moto 0
        [14.0, -0.3],  # Moto 1 - converging with 0
        [16.0, 0.0],   # Moto 2
        [15.5, 0.8],   # Moto 3 - converging with 0
        [13.0, 0.2],   # Moto 4
    ])
    
    logger.info("   Positions:")
    for i, pos in enumerate(positions):
        logger.info(f"     Moto {i}: ({pos[0]:.1f}, {pos[1]:.1f}) m")
    
    # Predict collision risks
    logger.info("\n[4] Predicting collision risks...")
    risks = safety_system.predict_collision_risk(positions, velocities)
    
    logger.info("   Collision Risks:")
    for agent_id, risk in risks.items():
        logger.info(f"     Moto {agent_id}: {risk:.3f} ({safety_system._classify_risk(risk)})")
    
    # Generate proximity alerts
    logger.info("\n[5] Generating proximity alerts...")
    alerts = safety_system.get_proximity_alerts(positions, velocities)
    
    logger.info("   Proximity Alerts:")
    for agent_id, alert in alerts.items():
        if alert['alert_active']:
            logger.info(f"     ⚠️ Moto {agent_id}: ALERT ACTIVE")
            logger.info(f"        Risk: {alert['risk']:.3f} ({alert['risk_level']})")
            logger.info(f"        Haptic: {alert['haptic_pattern']} @ {alert['haptic_intensity']:.1f}")
        else:
            logger.info(f"     ✓ Moto {agent_id}: Safe (risk={alert['risk']:.3f})")
    
    # Demonstrate reward modification
    logger.info("\n[6] Demonstrating RL reward modification...")
    base_reward = 1.0
    for agent_id, risk in risks.items():
        modified_reward = safety_system.compute_safety_reward(
            agent_id, base_reward, risk, penalty_weight=0.5
        )
        logger.info(f"   Moto {agent_id}: Base={base_reward:.2f} → Modified={modified_reward:.2f} (penalty={modified_reward-base_reward:.2f})")
    
    # Generate haptic patterns
    logger.info("\n[7] Haptic feedback patterns...")
    for pattern_type in ['rapid_pulse', 'slow_pulse', 'continuous']:
        pattern = generate_haptic_pattern(pattern_type)
        logger.info(f"   {pattern_type}: {pattern['frequency']}Hz, amplitude={pattern['amplitude']}")
    
    logger.info("\n" + "="*70)
    logger.info("✓ V2V Safety System Demo Complete")
    logger.info("="*70)
