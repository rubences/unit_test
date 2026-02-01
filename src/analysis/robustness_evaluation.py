"""
Robustness Evaluation & Visualization Script.

Genera gráficas de "Performance vs Noise Level" comparando:
- Baseline Model (trained sin ruido adversario)
- Adversarial-Trained Model (con curriculum learning)

Métricas:
- Mean Reward: Promedio de recompensas por episodio
- Success Rate: % episodios exitosos (reward > threshold)
- Robustness Score: Métrica compuesta que mide degradación

Visualización:
1. Performance curves (reward vs noise)
2. Success rates (completeness vs noise)
3. Robustness comparison (baseline vs adversarial)
4. Perturbation magnitude analysis
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_results(results_path: str) -> Dict[str, Any]:
    """Load results from JSON file.

    Args:
        results_path: Path to robustness_results.json

    Returns:
        Results dictionary
    """
    with open(results_path, "r") as f:
        return json.load(f)


def compute_robustness_score(
    baseline_rewards: List[float],
    adversarial_rewards: List[float],
    noise_levels: List[float],
) -> Tuple[float, Dict]:
    """Compute composite robustness score.

    Score combines:
    1. Improvement at max noise: How much better adversarial model is at 20% noise
    2. Consistency: How stable performance is across noise levels
    3. Relative degradation: How much baseline degrades vs adversarial

    Args:
        baseline_rewards: Mean rewards for baseline model at each noise level
        adversarial_rewards: Mean rewards for adversarial model
        noise_levels: Noise levels evaluated

    Returns:
        (robustness_score, metrics_dict)
    """
    baseline_rewards = np.array(baseline_rewards)
    adversarial_rewards = np.array(adversarial_rewards)

    # 1. Improvement at max noise (last point = hardest condition)
    improvement_at_max = (
        adversarial_rewards[-1] - baseline_rewards[-1]
    ) / (np.abs(baseline_rewards[-1]) + 1e-6)

    # 2. Consistency: inverse of standard deviation of degradation
    baseline_degradation = baseline_rewards[0] - baseline_rewards
    adversarial_degradation = adversarial_rewards[0] - adversarial_rewards
    consistency = 1.0 - (
        np.std(adversarial_degradation) / (np.std(baseline_degradation) + 1e-6)
    )
    consistency = max(0, consistency)

    # 3. Average relative improvement
    avg_improvement = np.mean(
        (adversarial_rewards - baseline_rewards) / (np.abs(baseline_rewards) + 1e-6)
    )

    # Combined score: weighted average
    robustness_score = (
        0.4 * improvement_at_max
        + 0.3 * consistency
        + 0.3 * avg_improvement
    )

    return robustness_score, {
        "improvement_at_max": float(improvement_at_max),
        "consistency": float(consistency),
        "avg_improvement": float(avg_improvement),
        "robustness_score": float(robustness_score),
    }


def plot_performance_comparison(
    results: Dict[str, Any],
    output_path: str = "robustness_comparison.png",
    figsize: Tuple[int, int] = (14, 10),
) -> None:
    """Create comprehensive robustness comparison plots.

    Args:
        results: Results dictionary from evaluation
        output_path: Where to save the figure
        figsize: Figure size (width, height)
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available, skipping plots")
        return

    noise_levels = np.array(results["noise_levels"]) * 100  # Convert to percentage

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Adversarial Training Robustness Evaluation", fontsize=16, fontweight="bold")

    models = results["models"]
    colors = {"Baseline": "#FF6B6B", "Adversarial": "#4ECDC4"}
    markers = {"Baseline": "o", "Adversarial": "s"}

    # ========== Plot 1: Mean Rewards vs Noise Level ==========
    ax = axes[0, 0]
    for model_name in ["Baseline", "Adversarial"]:
        if model_name not in models:
            continue

        rewards = models[model_name]["mean_rewards"]
        stds = models[model_name]["std_rewards"]

        ax.errorbar(
            noise_levels,
            rewards,
            yerr=stds,
            marker=markers[model_name],
            markersize=8,
            linestyle="-",
            linewidth=2.5,
            label=model_name,
            color=colors[model_name],
            capsize=5,
            capthick=2,
            alpha=0.8,
        )

    ax.set_xlabel("Sensor Noise Level (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Episode Reward", fontsize=11, fontweight="bold")
    ax.set_title("Performance Degradation Under Sensor Noise", fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xticks(noise_levels)

    # ========== Plot 2: Success Rate vs Noise Level ==========
    ax = axes[0, 1]
    for model_name in ["Baseline", "Adversarial"]:
        if model_name not in models:
            continue

        success_rates = np.array(models[model_name]["success_rates"]) * 100

        ax.plot(
            noise_levels,
            success_rates,
            marker=markers[model_name],
            markersize=8,
            linestyle="-",
            linewidth=2.5,
            label=model_name,
            color=colors[model_name],
            alpha=0.8,
        )

    ax.set_xlabel("Sensor Noise Level (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Success Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("Episode Success Rate vs Noise", fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim([0, 105])
    ax.set_xticks(noise_levels)

    # ========== Plot 3: Robustness Score Components ==========
    ax = axes[1, 0]
    baseline_rewards = np.array(models["Baseline"]["mean_rewards"])
    adversarial_rewards = np.array(models["Adversarial"]["mean_rewards"])
    noise_vals = np.array(results["noise_levels"])

    robustness_score, metrics = compute_robustness_score(
        baseline_rewards, adversarial_rewards, noise_vals
    )

    metric_names = [
        "Improvement\nat Max Noise",
        "Consistency",
        "Avg\nImprovement",
    ]
    metric_values = [
        metrics["improvement_at_max"],
        metrics["consistency"],
        metrics["avg_improvement"],
    ]
    metric_colors = ["#FFD93D", "#6BCB77", "#4D96FF"]

    bars = ax.bar(metric_names, metric_values, color=metric_colors, edgecolor="black", linewidth=2)

    # Add value labels on bars
    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title("Robustness Metrics", fontsize=12, fontweight="bold")
    ax.set_ylim([min(0, min(metric_values) - 0.1), max(metric_values) + 0.2])
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    # ========== Plot 4: Performance Degradation Rate ==========
    ax = axes[1, 1]

    # Calculate degradation rate (slope)
    baseline_rewards = np.array(models["Baseline"]["mean_rewards"])
    adversarial_rewards = np.array(models["Adversarial"]["mean_rewards"])

    # Normalize to [0, 1] for comparison
    baseline_norm = (baseline_rewards - np.min(baseline_rewards)) / (
        np.max(baseline_rewards) - np.min(baseline_rewards) + 1e-6
    )
    adversarial_norm = (adversarial_rewards - np.min(adversarial_rewards)) / (
        np.max(adversarial_rewards) - np.min(adversarial_rewards) + 1e-6
    )

    degradation_baseline = baseline_norm[0] - baseline_norm
    degradation_adversarial = adversarial_norm[0] - adversarial_norm

    width = 0.35
    x = np.arange(len(noise_levels))

    bars1 = ax.bar(
        x - width / 2,
        degradation_baseline * 100,
        width,
        label="Baseline",
        color=colors["Baseline"],
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        degradation_adversarial * 100,
        width,
        label="Adversarial",
        color=colors["Adversarial"],
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )

    ax.set_xlabel("Sensor Noise Level (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Performance Degradation (%)", fontsize=11, fontweight="bold")
    ax.set_title("Normalized Performance Loss", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(n)}%" for n in noise_levels])
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    # Add overall robustness score annotation
    textstr = f"Overall Robustness Score: {robustness_score:.3f}\n" \
              f"(Higher is better, max=1.0)"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    fig.text(0.99, 0.02, textstr, transform=fig.transFigure, fontsize=11,
            verticalalignment="bottom", horizontalalignment="right", bbox=props)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved comparison plot to {output_path}")
    plt.close()

    return robustness_score, metrics


def generate_robustness_report(
    results: Dict[str, Any],
    robustness_score: float,
    metrics: Dict[str, Any],
    output_path: str = "robustness_report.txt",
) -> None:
    """Generate detailed robustness evaluation report.

    Args:
        results: Results dictionary
        robustness_score: Computed robustness score
        metrics: Metrics dictionary
        output_path: Where to save report
    """
    report = []
    report.append("=" * 70)
    report.append("ADVERSARIAL TRAINING ROBUSTNESS EVALUATION REPORT")
    report.append("=" * 70)
    report.append("")

    # Noise levels tested
    noise_levels = results["noise_levels"]
    report.append("EVALUATION PARAMETERS")
    report.append("-" * 70)
    report.append(f"Noise Levels Tested: {[f'{n:.0%}' for n in noise_levels]}")
    report.append(f"Total Models Evaluated: {len(results['models'])}")
    report.append("")

    # Performance comparison
    report.append("PERFORMANCE COMPARISON")
    report.append("-" * 70)

    models = results["models"]
    for model_name in ["Baseline", "Adversarial"]:
        if model_name not in models:
            continue

        report.append(f"\n{model_name} Model:")
        report.append(f"  {'Noise':<8} {'Reward':<12} {'Success Rate':<15} {'Perturbation':<12}")
        report.append(f"  {'-'*8} {'-'*12} {'-'*15} {'-'*12}")

        for i, noise_level in enumerate(noise_levels):
            reward = models[model_name]["mean_rewards"][i]
            success = models[model_name]["success_rates"][i] * 100
            pert = models[model_name]["perturbations"][i]
            report.append(
                f"  {noise_level:5.0%}   {reward:9.3f}      {success:5.1f}%        {pert:8.4f}"
            )

    # Robustness analysis
    report.append("\n" + "=" * 70)
    report.append("ROBUSTNESS ANALYSIS")
    report.append("-" * 70)

    report.append(f"\nOverall Robustness Score: {robustness_score:.3f}")
    report.append("(Ranges from -1.0 to +1.0, higher is better)")
    report.append("")

    report.append("Component Scores:")
    report.append(f"  - Improvement at Max Noise: {metrics['improvement_at_max']:.3f}")
    report.append(f"  - Consistency: {metrics['consistency']:.3f}")
    report.append(f"  - Average Improvement: {metrics['avg_improvement']:.3f}")
    report.append("")

    # Key findings
    report.append("KEY FINDINGS")
    report.append("-" * 70)

    baseline_perf_0 = models["Baseline"]["mean_rewards"][0]
    baseline_perf_max = models["Baseline"]["mean_rewards"][-1]
    adversarial_perf_0 = models["Adversarial"]["mean_rewards"][0]
    adversarial_perf_max = models["Adversarial"]["mean_rewards"][-1]

    baseline_degradation = ((baseline_perf_max - baseline_perf_0) / abs(baseline_perf_0)) * 100
    adversarial_degradation = ((adversarial_perf_max - adversarial_perf_0) / abs(adversarial_perf_0)) * 100

    report.append(f"\nBaseline degradation (0% → max noise): {baseline_degradation:.1f}%")
    report.append(f"Adversarial degradation (0% → max noise): {adversarial_degradation:.1f}%")
    report.append(f"Improvement in robustness: {baseline_degradation - adversarial_degradation:.1f}%")
    report.append("")

    baseline_success_max = models["Baseline"]["success_rates"][-1] * 100
    adversarial_success_max = models["Adversarial"]["success_rates"][-1] * 100

    report.append(f"\nBaseline success rate at max noise: {baseline_success_max:.1f}%")
    report.append(f"Adversarial success rate at max noise: {adversarial_success_max:.1f}%")
    report.append("")

    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 70)

    if robustness_score > 0.3:
        report.append("✓ Adversarial training significantly improves robustness")
        report.append("✓ Model is suitable for deployment with sensor noise tolerance")
    elif robustness_score > 0.0:
        report.append("~ Adversarial training provides moderate robustness improvement")
        report.append("~ Consider extending training duration or curriculum stages")
    else:
        report.append("✗ Limited improvement from adversarial training")
        report.append("✗ Consider different attack strategies or longer training")

    report.append("")
    report.append("=" * 70)

    report_text = "\n".join(report)
    with open(output_path, "w") as f:
        f.write(report_text)

    logger.info(f"Saved report to {output_path}")
    print(report_text)


def main():
    """Main evaluation and visualization."""
    logger.info("ROBUSTNESS VISUALIZATION & ANALYSIS")
    logger.info("=" * 70)

    # Load results
    results_path = "models/adversarial/robustness_results.json"
    if not Path(results_path).exists():
        logger.error(f"Results file not found: {results_path}")
        logger.info("Run adversarial_training.py first to generate results")
        return

    logger.info(f"Loading results from {results_path}")
    results = load_results(results_path)

    # Create output directory
    output_dir = Path("models/adversarial")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    logger.info("\nGenerating comparison plots...")
    robustness_score, metrics = plot_performance_comparison(
        results,
        output_path=str(output_dir / "robustness_comparison.png"),
    )

    # Generate report
    logger.info("\nGenerating robustness report...")
    generate_robustness_report(
        results,
        robustness_score,
        metrics,
        output_path=str(output_dir / "robustness_report.txt"),
    )

    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Robustness Score: {robustness_score:.3f}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
