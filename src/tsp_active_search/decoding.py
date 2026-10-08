"""Greedy and multi-sample inference helpers."""

from __future__ import annotations

import numpy as np
import torch

from .policy import PointerPolicy
from .tsplib import TSPInstance


def _features(instance: TSPInstance, policy: PointerPolicy) -> torch.Tensor:
    parameter = next(policy.parameters())
    return torch.as_tensor(
        instance.node_features(), dtype=torch.float32, device=parameter.device
    ).unsqueeze(0)


@torch.no_grad()
def greedy_decode(instance: TSPInstance, policy: PointerPolicy) -> tuple[int, ...]:
    policy.eval()
    tour, _ = policy.sample_tours(_features(instance, policy), greedy=True)
    return tuple(int(node) for node in tour[0].cpu().numpy())


@torch.no_grad()
def sample_best(
    instance: TSPInstance, policy: PointerPolicy, sample_count: int = 128, seed: int | None = None
) -> tuple[int, ...]:
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")
    if seed is not None:
        torch.manual_seed(seed)
    policy.eval()
    features = _features(instance, policy).expand(sample_count, -1, -1)
    tours, _ = policy.sample_tours(features)
    tour_array = tours.cpu().numpy()
    lengths = [instance.tour_length(tour) for tour in tour_array]
    return tuple(int(node) for node in tour_array[int(np.argmin(lengths))])
