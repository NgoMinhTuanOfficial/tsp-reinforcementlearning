"""Plot tours and Active Search progress."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .active_search import ActiveSearchResult
from .tsplib import TSPInstance


def plot_tour(
    instance: TSPInstance, tour: tuple[int, ...], ax: plt.Axes | None = None
) -> tuple[Figure, plt.Axes]:
    if instance.coordinates is None or instance.coordinates.shape[1] < 2:
        raise ValueError("tour plotting requires at least two coordinates per node")
    if ax is None:
        figure, ax = plt.subplots(figsize=(7, 6))
    else:
        figure = ax.figure

    coords = instance.coordinates
    closed_tour = (*tour, tour[0])
    route = coords[np.asarray(closed_tour, dtype=np.int64)]
    ax.plot(route[:, 0], route[:, 1], "-o", linewidth=1.5, markersize=5)
    for index, (x, y) in enumerate(coords):
        ax.annotate(instance.node_ids[index], (x, y), xytext=(4, 4), textcoords="offset points")
    ax.set_title(f"{instance.name}: tour length {instance.tour_length(tour):g}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)
    return figure, ax


def plot_history(
    result: ActiveSearchResult, ax: plt.Axes | None = None
) -> tuple[Figure, plt.Axes]:
    if ax is None:
        figure, ax = plt.subplots(figsize=(7, 4))
    else:
        figure = ax.figure
    ax.plot(np.arange(1, len(result.history) + 1), result.history)
    ax.set_title("Best tour found during Active Search")
    ax.set_xlabel("Policy update")
    ax.set_ylabel("Best tour length")
    ax.grid(alpha=0.25)
    return figure, ax


def plot_result(instance: TSPInstance, result: ActiveSearchResult) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_tour(instance, result.best_tour, axes[0])
    plot_history(result, axes[1])
    figure.tight_layout()
    return figure
