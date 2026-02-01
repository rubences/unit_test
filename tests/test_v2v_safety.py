"""
Test Suite: V2V Safety System with GNN and Multi-Agent Environment

Tests:
1. GNN architecture and forward pass
2. Dynamic graph construction
3. V2V safety system collision prediction
4. Multi-agent environment dynamics
5. RL reward modulation
6. Haptic alert generation
7. End-to-end integration
"""

import pytest
import numpy as np
import torch
from torch_geometric.data import Data
import logging

from src.safety.gnn_v2v import (
    GNNPolicy,
    V2VGraph,
    V2VSafetySystem,
    generate_haptic_pattern
)
from src.environments.multi_moto_env import MultiMotoRacingEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestGNNPolicy:
    """Test GNN architecture."""
    
    def test_gnn_initialization(self):
        """Test GNN model creation."""
        model = GNNPolicy(node_features=4, hidden_dim=64, output_dim=1)
        assert model.node_features == 4
        assert model.hidden_dim == 64
        logger.info("✓ GNN initialization passed")
    
    def test_gnn_forward_pass(self):
        """Test GNN forward pass with sample data."""
        model = GNNPolicy(node_features=4, hidden_dim=64, output_dim=1)
        
        # Create sample graph (3 nodes)
        x = torch.randn(3, 4)  # 3 nodes, 4 features each
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        batch = torch.zeros(3, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, batch=batch)
        
        # Forward pass
        output = model(data)
        
        assert output.shape == (3, 1)
        assert torch.all((output >= 0) & (output <= 1))  # Sigmoid output
        logger.info(f"✓ GNN forward pass: input shape=(3,4), output shape={output.shape}")
    
    def test_gnn_parameter_count(self):
        """Test GNN has reasonable number of parameters."""
        model = GNNPolicy(node_features=4, hidden_dim=64, output_dim=1)
        param_count = sum(p.numel() for p in model.parameters())
        
        assert 5000 < param_count < 50000  # Reasonable range
        logger.info(f"✓ GNN parameter count: {param_count}")


class TestV2VGraph:
    """Test dynamic graph construction."""
    
    def test_graph_initialization(self):
        """Test V2VGraph creation."""
        graph = V2VGraph(proximity_threshold=10.0)
        assert graph.proximity_threshold == 10.0
        logger.info("✓ V2VGraph initialization passed")
    
    def test_graph_construction_close_agents(self):
        """Test graph construction when agents are close."""
        graph = V2VGraph(proximity_threshold=10.0)
        
        # 3 agents, all close to each other
        positions = np.array([[0.0, 0.0], [5.0, 0.0], [9.0, 0.0]])
        velocities = np.array([[15.0, 0.0], [14.0, 0.0], [16.0, 0.0]])
        
        data = graph.construct_graph(positions, velocities)
        
        assert data.num_nodes == 3
        assert data.x.shape == (3, 4)
        assert data.edge_index.shape[1] > 0  # Should have edges
        logger.info(f"✓ Graph with close agents: {data.edge_index.shape[1]} edges")
    
    def test_graph_construction_distant_agents(self):
        """Test graph construction when agents are far apart."""
        graph = V2VGraph(proximity_threshold=10.0)
        
        # 3 agents, all far from each other
        positions = np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]])
        velocities = np.array([[15.0, 0.0], [14.0, 0.0], [16.0, 0.0]])
        
        data = graph.construct_graph(positions, velocities)
        
        assert data.num_nodes == 3
        assert data.edge_index.shape[1] == 0  # No edges (too far)
        logger.info("✓ Graph with distant agents: 0 edges (as expected)")
    
    def test_graph_mixed_distances(self):
        """Test graph with mixed close/far agents."""
        graph = V2VGraph(proximity_threshold=10.0)
        
        # 4 agents: 0-1 close, 2-3 close, but groups far apart
        positions = np.array([[0.0, 0.0], [5.0, 0.0], [50.0, 0.0], [55.0, 0.0]])
        velocities = np.array([[15.0, 0.0]] * 4)
        
        data = graph.construct_graph(positions, velocities)
        
        assert data.num_nodes == 4
        # Should have 4 edges: 0↔1 and 2↔3
        assert data.edge_index.shape[1] == 4
        logger.info(f"✓ Mixed distance graph: {data.edge_index.shape[1]} edges")


class TestV2VSafetySystem:
    """Test V2V safety system integration."""
    
    def test_safety_system_initialization(self):
        """Test V2VSafetySystem creation."""
        gnn_model = GNNPolicy()
        safety = V2VSafetySystem(
            gnn_model=gnn_model,
            proximity_threshold=10.0,
            collision_threshold=0.7,
        )
        
        assert safety.collision_threshold == 0.7
        assert safety.haptic_alert_intensity == 0.9
        logger.info("✓ V2VSafetySystem initialization passed")
    
    def test_collision_risk_prediction(self):
        """Test collision risk prediction."""
        gnn_model = GNNPolicy()
        safety = V2VSafetySystem(gnn_model=gnn_model)
        
        positions = np.array([[0.0, 0.0], [5.0, 0.0], [20.0, 0.0]])
        velocities = np.array([[15.0, 0.0], [14.0, 0.0], [16.0, 0.0]])
        
        risks = safety.predict_collision_risk(positions, velocities)
        
        assert len(risks) == 3
        assert all(0 <= r <= 1 for r in risks.values())
        logger.info(f"✓ Collision risks: {[f'{r:.3f}' for r in risks.values()]}")
    
    def test_proximity_alerts(self):
        """Test proximity alert generation."""
        gnn_model = GNNPolicy()
        safety = V2VSafetySystem(gnn_model=gnn_model, collision_threshold=0.5)
        
        positions = np.array([[0.0, 0.0], [5.0, 0.0]])
        velocities = np.array([[15.0, 0.0], [15.0, 0.0]])
        
        alerts = safety.get_proximity_alerts(positions, velocities)
        
        assert len(alerts) == 2
        for agent_id, alert in alerts.items():
            assert 'risk' in alert
            assert 'alert_active' in alert
            assert 'haptic_pattern' in alert
        
        logger.info(f"✓ Proximity alerts generated for {len(alerts)} agents")
    
    def test_safety_reward_computation(self):
        """Test RL reward modification."""
        gnn_model = GNNPolicy()
        safety = V2VSafetySystem(gnn_model=gnn_model)
        
        base_reward = 1.0
        
        # Low risk → small penalty
        low_risk_reward = safety.compute_safety_reward(0, base_reward, 0.1)
        assert low_risk_reward > 0.9
        
        # High risk → large penalty
        high_risk_reward = safety.compute_safety_reward(0, base_reward, 0.9)
        assert high_risk_reward < 0.6
        
        logger.info(f"✓ Reward modulation: low_risk={low_risk_reward:.2f}, high_risk={high_risk_reward:.2f}")


class TestHapticPatterns:
    """Test haptic feedback generation."""
    
    def test_rapid_pulse_pattern(self):
        """Test rapid pulse pattern for imminent collision."""
        pattern = generate_haptic_pattern('rapid_pulse')
        
        assert pattern['frequency'] == 10.0
        assert pattern['amplitude'] == 0.9
        assert 'pulsing' in pattern['description'].lower()  # Check for "pulsing" or "pulse"
        logger.info("✓ Rapid pulse pattern generated")
    
    def test_all_pattern_types(self):
        """Test all haptic pattern types."""
        patterns = ['rapid_pulse', 'slow_pulse', 'continuous', 'none']
        
        for pattern_type in patterns:
            pattern = generate_haptic_pattern(pattern_type)
            assert 'frequency' in pattern
            assert 'amplitude' in pattern
            logger.info(f"  ✓ {pattern_type}: {pattern['frequency']}Hz, amplitude={pattern['amplitude']}")


class TestMultiMotoEnvironment:
    """Test multi-agent racing environment."""
    
    def test_environment_creation(self):
        """Test environment initialization."""
        env = MultiMotoRacingEnv(num_agents=5, enable_v2v=True)
        
        assert env._num_agents == 5
        assert len(env.possible_agents) == 5
        assert env.enable_v2v == True
        logger.info("✓ Environment created with 5 agents")
    
    def test_environment_reset(self):
        """Test environment reset."""
        env = MultiMotoRacingEnv(num_agents=3, enable_v2v=True)
        observations, infos = env.reset(seed=42)
        
        assert len(observations) == 3
        assert len(infos) == 3
        
        for agent in env.agents:
            assert agent in observations
            assert observations[agent].shape == (8,)
        
        logger.info(f"✓ Environment reset: {len(observations)} observations")
    
    def test_environment_step(self):
        """Test environment step."""
        env = MultiMotoRacingEnv(num_agents=3, enable_v2v=True)
        observations, _ = env.reset(seed=42)
        
        # Random actions
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        # Step
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        assert len(obs) == 3
        assert len(rewards) == 3
        assert len(infos) == 3
        
        logger.info(f"✓ Environment step: rewards={[f'{r:.2f}' for r in rewards.values()]}")
    
    def test_collision_risk_in_observations(self):
        """Test that collision risk is included in observations."""
        env = MultiMotoRacingEnv(num_agents=3, enable_v2v=True)
        observations, _ = env.reset(seed=42)
        
        for agent, obs in observations.items():
            collision_risk = obs[6]  # Index 6 is collision_risk
            assert 0 <= collision_risk <= 1
        
        logger.info("✓ Collision risk in observations")
    
    def test_proximity_alerts_in_infos(self):
        """Test that proximity alerts are in info dicts."""
        env = MultiMotoRacingEnv(num_agents=3, enable_v2v=True)
        observations, infos = env.reset(seed=42)
        
        for agent, info in infos.items():
            assert 'collision_risk' in info
            assert 'proximity_alert' in info
            assert 'haptic_pattern' in info
            assert 'haptic_intensity' in info
        
        logger.info("✓ Proximity alerts in info dicts")
    
    def test_episode_running(self):
        """Test running full episode."""
        env = MultiMotoRacingEnv(num_agents=3, enable_v2v=True, max_steps=50)
        observations, _ = env.reset(seed=42)
        
        total_rewards = {agent: 0.0 for agent in env.agents}
        
        for step in range(50):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, rewards, terms, truncs, infos = env.step(actions)
            
            for agent in env.agents:
                total_rewards[agent] += rewards[agent]
        
        logger.info(f"✓ Episode complete: total_rewards={[f'{r:.1f}' for r in total_rewards.values()]}")


class TestIntegration:
    """End-to-end integration tests."""
    
    def test_gnn_to_environment_integration(self):
        """Test complete GNN → Environment → RL pipeline."""
        logger.info("\n" + "="*60)
        logger.info("INTEGRATION TEST: GNN → Environment → RL")
        logger.info("="*60)
        
        # Create environment
        logger.info("\n[1/4] Creating environment...")
        env = MultiMotoRacingEnv(num_agents=3, enable_v2v=True, max_steps=20)
        logger.info("  ✓ Environment created")
        
        # Reset
        logger.info("\n[2/4] Resetting environment...")
        observations, infos = env.reset(seed=42)
        logger.info(f"  ✓ Reset complete: {len(observations)} agents")
        
        # Run episode and track V2V activity
        logger.info("\n[3/4] Running episode with V2V safety...")
        alert_counts = {agent: 0 for agent in env.agents}
        risk_levels = {agent: [] for agent in env.agents}
        
        for step in range(20):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, rewards, terms, truncs, infos = env.step(actions)
            
            for agent in env.agents:
                if infos[agent]['proximity_alert']:
                    alert_counts[agent] += 1
                risk_levels[agent].append(infos[agent]['collision_risk'])
        
        logger.info("  ✓ Episode complete")
        
        # Analyze results
        logger.info("\n[4/4] Analyzing V2V safety results...")
        for agent in env.agents:
            avg_risk = np.mean(risk_levels[agent])
            max_risk = np.max(risk_levels[agent])
            alerts = alert_counts[agent]
            logger.info(f"  {agent}: avg_risk={avg_risk:.3f}, max_risk={max_risk:.3f}, alerts={alerts}")
        
        logger.info("\n" + "="*60)
        logger.info("✓ INTEGRATION TEST PASSED")
        logger.info("="*60)


if __name__ == '__main__':
    logger.info("\n" + "="*70)
    logger.info("V2V SAFETY SYSTEM - TEST SUITE")
    logger.info("="*70)
    
    # Run tests
    pytest.main([__file__, '-v', '--tb=short', '--disable-warnings'])
