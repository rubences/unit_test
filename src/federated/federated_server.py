"""
Federated Learning Server using Flower (flwr) with FedAvg aggregation strategy.

This server coordinates training across multiple clients (pilots) without
centralizing their telemetry data, preserving privacy via federated averaging.

Usage:
    python src/federated/federated_server.py --rounds 5 --min-available-clients 3
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

try:
    import flwr as fl
    FLWR_AVAILABLE = True
except ImportError:
    FLWR_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class CustomFedAvg(fl.server.strategy.FedAvg):
    """Custom FedAvg strategy with loss tracking for convergence analysis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_history = []
        self.round = 0

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ) -> Tuple:
        """Aggregate fit results and track loss."""
        aggregated_weights, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_metrics:
            avg_loss = np.mean([m.get("loss", 0.0) for _, m in results])
            self.loss_history.append(avg_loss)
            logger.info(f"Round {server_round}: Avg loss = {avg_loss:.4f}")

        self.round = server_round
        return aggregated_weights, aggregated_metrics

    def save_metrics(self, output_path: Path) -> None:
        """Save loss history to JSON for plotting."""
        metrics = {
            "rounds": list(range(1, len(self.loss_history) + 1)),
            "loss": self.loss_history,
            "strategy": "FedAvg",
            "type": "federated",
        }
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {output_path}")


def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Compute weighted average of metrics across clients."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_evaluate_fn(num_rounds: int):
    """Return an evaluation function for the global model."""

    def evaluate(
        server_round: int,
        parameters,
        config,
    ) -> Tuple:
        """Evaluate global model (placeholder)."""
        logger.info(f"Evaluating global model at round {server_round}/{num_rounds}")
        return 0.0, {"accuracy": 0.5}  # Placeholder

    return evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated Learning Server (Flower)")
    parser.add_argument("--host", default="localhost", help="Server host address")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--rounds", type=int, default=5, help="Number of FL rounds")
    parser.add_argument("--min-available-clients", type=int, default=3, help="Min clients to start")
    parser.add_argument("--output", type=Path, default="outputs/federated_metrics.json", help="Output metrics path")
    args = parser.parse_args()

    logger.info(f"Starting Federated Learning Server")
    logger.info(f"  Rounds: {args.rounds}")
    logger.info(f"  Min clients: {args.min_available_clients}")
    logger.info(f"  Address: {args.host}:{args.port}")

    strategy = CustomFedAvg(
        min_fit_clients=args.min_available_clients,
        min_evaluate_clients=args.min_available_clients,
        min_available_clients=args.min_available_clients,
        evaluate_fn=get_evaluate_fn(args.rounds),
        on_fit_config_fn=lambda r: {"lr": 0.001, "epochs": 5},
    )

    fl.server.start_server(
        server_address=f"{args.host}:{args.port}",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    strategy.save_metrics(args.output)
    logger.info("Server training complete")


if __name__ == "__main__":
    main()
