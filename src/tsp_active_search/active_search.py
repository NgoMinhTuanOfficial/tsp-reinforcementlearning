"""Per-instance Active Search optimization loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .policy import PointerPolicy
from .tsplib import TSPInstance


@dataclass(frozen=True)
class ActiveSearchConfig:
    iterations: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    baseline_alpha: float = 0.9
    hidden_dim: int = 128
    seed: int | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.baseline_alpha < 1:
            raise ValueError("baseline_alpha must be in [0, 1)")


@dataclass(frozen=True)
class ActiveSearchResult:
    best_tour: tuple[int, ...]
    best_length: float
    history: tuple[float, ...]
    policy: PointerPolicy


def _resolve_device(requested: str | None) -> torch.device:
    if requested is not None:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def active_search(
    instance: TSPInstance, config: ActiveSearchConfig | None = None
) -> ActiveSearchResult:
    """Optimize a fresh random policy for one instance; no pretraining is performed."""
    config = config or ActiveSearchConfig()
    if config.seed is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    device = _resolve_device(config.device)
    features = torch.as_tensor(
        instance.node_features(), dtype=torch.float32, device=device
    ).unsqueeze(0)
    policy = PointerPolicy(features.shape[-1], config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    initial_tour = np.random.permutation(instance.dimension)
    best_tour = tuple(int(node) for node in initial_tour)
    best_length = instance.tour_length(best_tour)
    baseline: float | None = None
    history: list[float] = []

    policy.train()
    for _ in range(config.iterations):
        optimizer.zero_grad(set_to_none=True)
        batch_features = features.expand(config.batch_size, -1, -1)
        tours, log_probabilities = policy.sample_tours(batch_features)
        tour_array = tours.detach().cpu().numpy()
        lengths_np = np.asarray(
            [instance.tour_length(tour) for tour in tour_array], dtype=np.float32
        )
        lengths = torch.as_tensor(lengths_np, dtype=torch.float32, device=device)

        candidate_index = int(lengths_np.argmin())
        if lengths_np[candidate_index] < best_length:
            best_tour = tuple(int(node) for node in tour_array[candidate_index])
            best_length = float(lengths_np[candidate_index])

        batch_mean = float(lengths_np.mean())
        if baseline is None:
            baseline = batch_mean
        advantage = lengths - baseline
        loss = (advantage.detach() * log_probabilities).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=2.0)
        optimizer.step()
        baseline = (
            config.baseline_alpha * baseline
            + (1.0 - config.baseline_alpha) * batch_mean
        )
        history.append(best_length)

    return ActiveSearchResult(best_tour, best_length, tuple(history), policy)
