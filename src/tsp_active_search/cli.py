"""Command-line interface."""

from __future__ import annotations

import argparse

from .active_search import ActiveSearchConfig, active_search
from .tsplib import load_tsplib


def main() -> None:
    parser = argparse.ArgumentParser(description="Run per-instance TSP Active Search.")
    parser.add_argument("instance", help="path to a TSPLIB .tsp file")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda")
    args = parser.parse_args()

    instance = load_tsplib(args.instance)
    result = active_search(
        instance,
        ActiveSearchConfig(
            iterations=args.iterations,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
        ),
    )
    labels = [instance.node_ids[index] for index in result.best_tour]
    print(f"Instance: {instance.name} ({instance.dimension} cities)")
    print(f"Best tour length: {result.best_length:g}")
    print(f"Best tour node labels: {' -> '.join(labels)} -> {labels[0]}")


if __name__ == "__main__":
    main()
