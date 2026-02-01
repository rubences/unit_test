#!/usr/bin/env python3
"""
Quick Reference: Adversarial Training API

Guía de uso rápido del módulo de entrenamiento adversario.
"""

# ============================================================================
# 1. CREAR AGENTE ADVERSARIO
# ============================================================================

from src.agents.sensor_noise_agent import SensorNoiseAgent, AdversarialEnvironmentWrapper
import numpy as np

# Opción A: Agente simple
agent = SensorNoiseAgent(noise_level=0.15)

# Opción B: Con configuración completa
agent = SensorNoiseAgent(
    noise_level=0.20,              # 20% sensor noise
    curriculum_stage=2,             # Medium difficulty
    attack_modes=["gaussian", "drift", "cutout", "bias"],
    seed=42,
)

# ============================================================================
# 2. INYECTAR RUIDO EN TELEMETRÍA
# ============================================================================

telemetry = np.array([1.2, 0.5, 9.8, 10.0, 2.5, 5.0])  # IMU: accel + gyro

# Aplicar ataque
corrupted_telemetry, metadata = agent.inject_noise(telemetry)

# Inspeccionar resultados
print(f"Attacks applied: {metadata['attacks_applied']}")
print(f"Perturbation magnitude: {metadata['perturbation_magnitude']:.4f}")

# ============================================================================
# 3. INTEGRACIÓN CON GYMNASIUM
# ============================================================================

import gymnasium as gym

# Crear ambiente
env = gym.make("CartPole-v1")

# Envolver con ruido adversario
adversarial_env = AdversarialEnvironmentWrapper(env, sensor_noise_agent=agent)

# Usar como ambiente normal
obs, info = adversarial_env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = adversarial_env.step(action)
    
    # Acceder a metadata de ataques
    if "adversarial" in info:
        print(f"Perturbation: {info['adversarial']['perturbation_magnitude']}")
    
    if terminated or truncated:
        break

# Obtener estadísticas del episodio
stats = adversarial_env.get_episode_stats()
print(f"Episode perturbation: {stats['avg_perturbation']:.4f}")

# ============================================================================
# 4. CURRICULUM LEARNING
# ============================================================================

# Cambiar intensidad de ataques
for stage in [1, 2, 3]:
    agent.set_curriculum_stage(stage)
    print(f"Stage {stage}: {agent.stage_params}")

# Cambiar nivel de ruido
agent.set_curriculum_stage(3)  # Hard stage
agent.noise_level = 0.25       # 25% maximum noise

# ============================================================================
# 5. ENTRENAR MODELO ADVERSARIO
# ============================================================================

from src.training.adversarial_training import TrainingConfig, train_adversarial

config = TrainingConfig(
    total_timesteps=50_000,
    stage_duration=10_000,
    curriculum_enabled=True,
    max_noise_level=0.20,
    eval_noise_levels=[0.0, 0.05, 0.10, 0.15, 0.20],
    eval_episodes=5,
)

adversarial_model, info = train_adversarial(config)

# ============================================================================
# 6. EVALUAR ROBUSTEZ
# ============================================================================

from src.training.adversarial_training import evaluate_robustness

models = {"My_Model": adversarial_model}
results = evaluate_robustness(models, config)

# Inspeccionar resultados
for model_name, model_data in results["models"].items():
    print(f"\n{model_name}:")
    for noise_level, reward in zip(results["noise_levels"], 
                                    model_data["mean_rewards"]):
        print(f"  {noise_level:.0%} noise: {reward:.3f}")

# ============================================================================
# 7. VISUALIZAR RESULTADOS
# ============================================================================

from src.analysis.robustness_evaluation import (
    plot_performance_comparison,
    generate_robustness_report,
    compute_robustness_score,
)

# Generar gráficas
robustness_score, metrics = plot_performance_comparison(
    results,
    output_path="robustness_comparison.png"
)

# Generar reporte
generate_robustness_report(results, robustness_score, metrics)

# ============================================================================
# 8. PERSONALIZAR ESTRATEGIAS DE ATAQUE
# ============================================================================

# Crear agente custom
class CustomAttackAgent(SensorNoiseAgent):
    """Agente con estrategia de ataque personalizada."""
    
    def inject_noise(self, telemetry):
        """Implementar ataque personalizado."""
        corrupted = telemetry.copy()
        
        # Custom: EMP pulse cada 100 steps
        if self.step_count % 100 == 0:
            corrupted[3:] = 1000.0  # Gyro spike
        
        # Custom: Temperatura degradation
        temp_factor = 1.0 + 0.01 * (self.step_count % 1000) / 1000
        corrupted *= temp_factor
        
        self.step_count += 1
        
        return corrupted, {
            "noise_level": self.noise_level,
            "curriculum_stage": self.curriculum_stage,
            "attacks_applied": ["emp_pulse", "temp_degradation"],
            "perturbation_magnitude": float(np.linalg.norm(corrupted - telemetry)),
        }

# Usar custom agent
custom_agent = CustomAttackAgent(noise_level=0.15)
env = gym.make("CartPole-v1")
adversarial_env = AdversarialEnvironmentWrapper(env, sensor_noise_agent=custom_agent)

# ============================================================================
# 9. DEBUGGING: INSPECCIONAR ESTADO
# ============================================================================

agent = SensorNoiseAgent(noise_level=0.20, curriculum_stage=2)

# Ver configuración
status = agent.get_status()
print(f"Attack strength: {status['attack_strength']:.3f}")
print(f"Stage params: {status['stage_params']}")

# Simular ataques múltiples
for i in range(5):
    tel = np.random.randn(6).astype(np.float32)
    corr, meta = agent.inject_noise(tel)
    print(f"Step {i+1}: attacks={meta['attacks_applied']}, "
          f"pert={meta['perturbation_magnitude']:.4f}")

# Reset antes de nuevo episodio
agent.reset_drift()

# ============================================================================
# 10. CASOS DE USO AVANZADOS
# ============================================================================

# A. Entrenamiento con ruido variable
noise_schedule = [0.0, 0.05, 0.10, 0.15, 0.20]
for epoch, noise in enumerate(noise_schedule):
    agent.noise_level = noise
    # ... entrenar epoch ...
    print(f"Epoch {epoch}: noise={noise:.0%}")

# B. Evaluación robustez gradual
for noise_level in np.linspace(0, 0.30, 7):
    agent.noise_level = noise_level
    # ... evaluar con este ruido ...
    print(f"Robustness at {noise_level:.0%}: {agent.get_attack_strength():.3f}")

# C. Análisis de sensibilidad
baseline_rewards = []
for sensor_idx in range(6):
    agent_single = SensorNoiseAgent(
        attack_modes=["gaussian"],
        noise_level=0.20,
    )
    # Attack single sensor
    # ... medir performance degradation ...
    print(f"Sensor {sensor_idx} sensitivity: ...")

# ============================================================================
# 11. TESTING RÁPIDO
# ============================================================================

def quick_test():
    """Test rápido del módulo."""
    import sys
    
    try:
        # Test 1: Crear agente
        agent = SensorNoiseAgent(noise_level=0.10)
        assert agent.noise_level == 0.10
        print("✓ Test 1 passed: Agent creation")
        
        # Test 2: Inyectar ruido
        tel = np.ones(6, dtype=np.float32)
        corr, meta = agent.inject_noise(tel)
        assert not np.allclose(corr, tel)
        print("✓ Test 2 passed: Noise injection")
        
        # Test 3: Curriculum
        agent.set_curriculum_stage(3)
        assert agent.curriculum_stage == 3
        print("✓ Test 3 passed: Curriculum progression")
        
        # Test 4: Wrapper
        env = gym.make("CartPole-v1")
        wrapper = AdversarialEnvironmentWrapper(env)
        obs, info = wrapper.reset()
        assert obs.shape[0] == 4
        print("✓ Test 4 passed: Wrapper integration")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    quick_test()

# ============================================================================
# 12. COMANDOS ÚTILES
# ============================================================================

"""
# Ejecutar tests completos
python -m pytest tests/test_adversarial_training.py -v

# Ejecutar demo
python scripts/adversarial_training_demo.py

# Entrenar modelo
python -m src.training.adversarial_training

# Generar visualizaciones
python -m src.analysis.robustness_evaluation

# Ver archivo de guía completa
cat docs/ADVERSARIAL_TRAINING_GUIDE.md

# Ver resumen ejecutivo
cat ADVERSARIAL_TRAINING_README.md
"""

# ============================================================================
