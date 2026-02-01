#!/usr/bin/env python3
"""
Quick Start Guide: Adversarial Training Demo

Este script demuestra el flujo completo de entrenamiento adversario:
1. Crear el agente villano con curriculum learning
2. Entrenar modelo baseline (sin ruido)
3. Entrenar modelo adversarial (con ruido progresivo)
4. Evaluar robustez en diferentes niveles de ruido
5. Generar gráficas comparativas

Requisitos:
- numpy
- stable-baselines3
- gymnasium
- matplotlib
"""

import logging
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.sensor_noise_agent import SensorNoiseAgent, AdversarialEnvironmentWrapper
from src.training.adversarial_training import TrainingConfig, train_baseline, train_adversarial, evaluate_robustness
from src.analysis.robustness_evaluation import (
    load_results,
    plot_performance_comparison,
    generate_robustness_report,
    compute_robustness_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def demo_sensor_noise_agent():
    """Demonstrate SensorNoiseAgent capabilities."""
    logger.info("\n" + "="*70)
    logger.info("DEMO 1: Sensor Noise Agent")
    logger.info("="*70)

    # Create agent
    agent = SensorNoiseAgent(
        noise_level=0.10,  # 10% noise
        curriculum_stage=1,  # Easy stage
        attack_modes=["gaussian", "drift", "cutout", "bias"],
    )

    logger.info("\nInitial Configuration:")
    logger.info(f"  {agent.get_status()}")

    # Simulate telemetry
    telemetry = [
        [1.2, 0.5, 9.8, 10.0, 2.5, 5.0],  # Normal reading
        [1.2, 0.5, 9.8, 10.0, 2.5, 5.0],  # Normal reading
        [1.2, 0.5, 9.8, 10.0, 2.5, 5.0],  # Normal reading
    ]

    logger.info("\nApplying attacks across 3 timesteps:")
    logger.info(f"  Clean telemetry: {telemetry[0]}")

    for i, tel in enumerate(telemetry, 1):
        import numpy as np
        corrupted, metadata = agent.inject_noise(np.array(tel))
        logger.info(f"\n  Step {i}:")
        logger.info(f"    Attacks: {metadata['attacks_applied']}")
        logger.info(f"    Perturbation magnitude: {metadata['perturbation_magnitude']:.4f}")
        logger.info(f"    Corrupted telemetry: {corrupted}")

    # Demonstrate curriculum progression
    logger.info("\n\nCurriculum Progression:")
    for stage in [1, 2, 3]:
        agent.set_curriculum_stage(stage)
        logger.info(f"  Stage {stage}: {agent.stage_params}")


def demo_curriculum_learning():
    """Demonstrate curriculum learning schedule."""
    logger.info("\n" + "="*70)
    logger.info("DEMO 2: Curriculum Learning Schedule")
    logger.info("="*70)

    logger.info("\nCurriculum Training Schedule:")
    logger.info("  Epochs 1-3:   Stage 1 (Easy)   - σ=0.1,   p_cutout=5%,  drift=0.001")
    logger.info("  Epochs 4-6:   Stage 2 (Medium) - σ=0.3,   p_cutout=15%, drift=0.005")
    logger.info("  Epochs 7-10:  Stage 3 (Hard)   - σ=0.5,   p_cutout=30%, drift=0.01")

    logger.info("\nExpected Learning Dynamics:")
    logger.info("  - Fase Early: Model ajusta a ruido débil (aprende rápido)")
    logger.info("  - Fase Medium: Model generaliza a ruido moderado")
    logger.info("  - Fase Hard: Model desarrolla robustez contra ataques agresivos")
    logger.info("  → Result: Better performance on clean AND noisy data!")


def run_full_pipeline(quick_mode: bool = True):
    """Run complete adversarial training pipeline.

    Args:
        quick_mode: If True, use reduced timesteps for faster demo
    """
    logger.info("\n" + "="*70)
    logger.info("DEMO 3: Full Adversarial Training Pipeline")
    logger.info("="*70)

    # Configuration
    if quick_mode:
        config = TrainingConfig(
            total_timesteps=10_000,  # Quick demo
            stage_duration=3_000,
            eval_episodes=3,
        )
        logger.info("Running in QUICK MODE (reduced timesteps)")
    else:
        config = TrainingConfig(
            total_timesteps=50_000,  # Full training
            stage_duration=10_000,
            eval_episodes=10,
        )
        logger.info("Running in FULL MODE")

    # Create output directory
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)

    # Train baseline
    logger.info("\n[1/3] Training baseline model...")
    baseline_model, baseline_info = train_baseline(config)
    logger.info(f"✓ Baseline training complete")

    # Train adversarial
    logger.info("\n[2/3] Training adversarial model with curriculum...")
    adversarial_model, adversarial_info = train_adversarial(config)
    logger.info(f"✓ Adversarial training complete")

    # Evaluate
    logger.info("\n[3/3] Evaluating robustness...")
    models = {
        "Baseline": baseline_model,
        "Adversarial": adversarial_model,
    }
    results = evaluate_robustness(models, config)
    logger.info(f"✓ Evaluation complete")

    # Print summary
    logger.info("\n" + "="*70)
    logger.info("RESULTS SUMMARY")
    logger.info("="*70)

    for model_name, model_data in results["models"].items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  Noise Level  | Reward       | Success Rate")
        logger.info(f"  {'='*45}")
        for i, noise_level in enumerate(results["noise_levels"]):
            reward = model_data["mean_rewards"][i]
            success = model_data["success_rates"][i] * 100
            logger.info(
                f"  {noise_level:5.0%}        | {reward:8.3f}     | {success:5.1f}%"
            )

    # Compute robustness score
    baseline_rewards = results["models"]["Baseline"]["mean_rewards"]
    adversarial_rewards = results["models"]["Adversarial"]["mean_rewards"]
    import numpy as np
    robustness_score, metrics = compute_robustness_score(
        baseline_rewards, adversarial_rewards, results["noise_levels"]
    )

    logger.info(f"\n{'='*45}")
    logger.info(f"Robustness Score: {robustness_score:.3f}")
    logger.info(f"  - Improvement at Max Noise: {metrics['improvement_at_max']:.3f}")
    logger.info(f"  - Consistency: {metrics['consistency']:.3f}")
    logger.info(f"  - Avg Improvement: {metrics['avg_improvement']:.3f}")

    return results, robustness_score, metrics


def main():
    """Run all demonstrations."""
    logger.info("\n" + "#"*70)
    logger.info("# ADVERSARIAL TRAINING FOR ROBUST MOTORCYCLE COACHING")
    logger.info("# Security Research: AI Model Robustness Evaluation")
    logger.info("#"*70)

    try:
        # Demo 1: Sensor Noise Agent
        demo_sensor_noise_agent()

        # Demo 2: Curriculum Learning
        demo_curriculum_learning()

        # Demo 3: Full Pipeline (in quick mode)
        try:
            results, robustness_score, metrics = run_full_pipeline(quick_mode=True)

            # Generate visualizations
            logger.info("\nGenerating visualizations...")
            output_path = "models/adversarial/robustness_comparison.png"
            plot_performance_comparison(results, output_path=output_path)
            logger.info(f"✓ Saved plot to {output_path}")

            # Generate report
            report_path = "models/adversarial/robustness_report.txt"
            generate_robustness_report(results, robustness_score, metrics, output_path=report_path)
            logger.info(f"✓ Saved report to {report_path}")

        except Exception as e:
            logger.warning(f"Pipeline execution failed (expected if dependencies missing): {e}")
            logger.info("This is normal - the pipeline requires specific dependencies")

        logger.info("\n" + "#"*70)
        logger.info("DEMO COMPLETE")
        logger.info("#"*70)
        logger.info("\nNext steps:")
        logger.info("1. Run full training: python -m src.training.adversarial_training")
        logger.info("2. Generate visualizations: python -m src.analysis.robustness_evaluation")
        logger.info("3. View results in models/adversarial/")

    except Exception as e:
        logger.error(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
