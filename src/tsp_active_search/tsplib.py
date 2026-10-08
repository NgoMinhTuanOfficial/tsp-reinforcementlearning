"""TSPLIB instance parsing and edge-weight calculation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TSPInstance:
    name: str
    node_ids: tuple[str, ...]
    coordinates: np.ndarray | None
    distance_matrix: np.ndarray
    edge_weight_type: str

    @property
    def dimension(self) -> int:
        return len(self.node_ids)

    def tour_length(self, tour: Iterable[int]) -> float:
        order = np.asarray(tuple(tour), dtype=np.int64)
        if order.shape != (self.dimension,):
            raise ValueError(f"tour must contain exactly {self.dimension} nodes")
        if not np.array_equal(np.sort(order), np.arange(self.dimension)):
            raise ValueError("tour must be a permutation of all node indices")
        return float(self.distance_matrix[order, np.roll(order, -1)].sum())

    def node_features(self) -> np.ndarray:
        """Return fixed-size per-node features suitable for the policy network."""
        if self.coordinates is not None:
            coords = self.coordinates.astype(np.float32, copy=True)
            span = np.ptp(coords, axis=0)
            scale = np.where(span > 0, span, 1.0)
            return (coords - coords.mean(axis=0, keepdims=True)) / scale

        matrix = self.distance_matrix.astype(np.float32, copy=True)
        scale = max(float(matrix.max()), 1.0)
        return matrix / scale


def _distance(a: np.ndarray, b: np.ndarray, weight_type: str) -> float:
    delta = np.abs(a - b)
    if weight_type in {"EUC_2D", "EUC_3D"}:
        return float(math.floor(float(np.linalg.norm(delta)) + 0.5))
    if weight_type == "CEIL_2D":
        return float(math.ceil(float(np.linalg.norm(delta))))
    if weight_type == "MAN_2D":
        return float(math.floor(float(delta.sum()) + 0.5))
    if weight_type == "MAX_2D":
        return float(math.floor(float(delta.max()) + 0.5))
    if weight_type == "ATT":
        raw = math.sqrt(float(np.square(delta).sum()) / 10.0)
        rounded = math.floor(raw + 0.5)
        return float(rounded + 1 if rounded < raw else rounded)
    if weight_type == "GEO":
        def to_radians(value: float) -> float:
            degrees = int(value)
            minutes = value - degrees
            return math.pi * (degrees + 5.0 * minutes / 3.0) / 180.0

        latitude_a, longitude_a = map(to_radians, a[:2])
        latitude_b, longitude_b = map(to_radians, b[:2])
        radius = 6378.388
        q1 = math.cos(longitude_a - longitude_b)
        q2 = math.cos(latitude_a - latitude_b)
        q3 = math.cos(latitude_a + latitude_b)
        value = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)
        return float(math.floor(radius * math.acos(max(-1.0, min(1.0, value))) + 1.0))
    raise ValueError(f"unsupported EDGE_WEIGHT_TYPE: {weight_type}")


def _explicit_matrix(values: list[float], dimension: int, fmt: str) -> np.ndarray:
    matrix = np.zeros((dimension, dimension), dtype=np.float64)
    idx = 0

    def take() -> float:
        nonlocal idx
        if idx >= len(values):
            raise ValueError("EDGE_WEIGHT_SECTION has too few values")
        value = values[idx]
        idx += 1
        return value

    if fmt == "FULL_MATRIX":
        if len(values) != dimension * dimension:
            raise ValueError("FULL_MATRIX must contain DIMENSION * DIMENSION values")
        return np.asarray(values, dtype=np.float64).reshape(dimension, dimension)

    layouts = {
        "UPPER_ROW": ("row", "upper", False),
        "LOWER_ROW": ("row", "lower", False),
        "UPPER_DIAG_ROW": ("row", "upper", True),
        "LOWER_DIAG_ROW": ("row", "lower", True),
        "UPPER_COL": ("col", "upper", False),
        "LOWER_COL": ("col", "lower", False),
        "UPPER_DIAG_COL": ("col", "upper", True),
        "LOWER_DIAG_COL": ("col", "lower", True),
    }
    if fmt not in layouts:
        raise ValueError(f"unsupported EDGE_WEIGHT_FORMAT: {fmt}")

    traversal, triangle, diagonal = layouts[fmt]
    if traversal == "row":
        outer_indices = range(dimension)
        pairs = (
            (i, j)
            for i in outer_indices
            for j in (
                range(i if diagonal else i + 1, dimension)
                if triangle == "upper"
                else range(0, i + 1 if diagonal else i)
            )
        )
    else:
        pairs = (
            (i, j)
            for j in range(dimension)
            for i in (
                range(0, j + 1 if diagonal else j)
                if triangle == "upper"
                else range(j if diagonal else j + 1, dimension)
            )
        )

    for i, j in pairs:
        value = take()
        matrix[i, j] = value
        matrix[j, i] = value
    if idx != len(values):
        raise ValueError("EDGE_WEIGHT_SECTION has too many values")
    return matrix


def load_tsplib(path: str | Path) -> TSPInstance:
    """Load a coordinate or explicit-matrix TSPLIB TSP instance."""
    source = Path(path)
    headers: dict[str, str] = {}
    coordinate_rows: list[tuple[str, list[float]]] = []
    display_rows: list[tuple[str, list[float]]] = []
    edge_values: list[float] = []
    section: str | None = None

    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("COMMENT"):
            continue
        if line == "EOF":
            break
        if line in {"NODE_COORD_SECTION", "DISPLAY_DATA_SECTION", "EDGE_WEIGHT_SECTION"}:
            section = line
            continue
        if section == "NODE_COORD_SECTION":
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"invalid coordinate row: {line}")
            coordinate_rows.append((parts[0], [float(value) for value in parts[1:]]))
            continue
        if section == "DISPLAY_DATA_SECTION":
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"invalid display-data row: {line}")
            display_rows.append((parts[0], [float(value) for value in parts[1:]]))
            continue
        if section == "EDGE_WEIGHT_SECTION":
            edge_values.extend(float(value) for value in line.split())
            continue

        if ":" in line:
            key, value = line.split(":", 1)
        else:
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            key, value = parts
        headers[key.strip().upper()] = value.strip()

    if "DIMENSION" not in headers:
        raise ValueError("TSPLIB file is missing DIMENSION")
    dimension = int(headers["DIMENSION"])
    if dimension < 2:
        raise ValueError("DIMENSION must be at least 2")
    weight_type = headers.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()
    name = headers.get("NAME", source.stem)
    coordinates: np.ndarray | None = None

    if weight_type == "EXPLICIT":
        if not edge_values:
            raise ValueError("EXPLICIT instance is missing EDGE_WEIGHT_SECTION")
        fmt = headers.get("EDGE_WEIGHT_FORMAT", "").upper()
        if not fmt:
            raise ValueError("EXPLICIT instance is missing EDGE_WEIGHT_FORMAT")
        distance_matrix = _explicit_matrix(edge_values, dimension, fmt)
        node_ids = tuple(str(i + 1) for i in range(dimension))
        display = coordinate_rows or display_rows
        if display:
            if len(display) != dimension:
                raise ValueError("display coordinates must match DIMENSION")
            node_ids = tuple(row[0] for row in display)
            coordinates = np.asarray([row[1] for row in display], dtype=np.float64)
            if len(set(node_ids)) != dimension:
                raise ValueError("node identifiers must be unique")
            if coordinates.ndim != 2 or coordinates.shape[1] < 2:
                raise ValueError("display coordinates must have at least two dimensions")
            if not np.isfinite(coordinates).all():
                raise ValueError("display coordinates must be finite")
    else:
        if not coordinate_rows:
            raise ValueError(f"{weight_type} instance is missing NODE_COORD_SECTION")
        if len(coordinate_rows) != dimension:
            raise ValueError(
                f"DIMENSION is {dimension}, but found {len(coordinate_rows)} coordinate rows"
            )
        node_ids = tuple(row[0] for row in coordinate_rows)
        coordinates = np.asarray([row[1] for row in coordinate_rows], dtype=np.float64)
        if len(set(node_ids)) != dimension:
            raise ValueError("node identifiers must be unique")
        expected_dimensions = 3 if weight_type == "EUC_3D" else 2
        if coordinates.shape != (dimension, expected_dimensions) or not np.isfinite(coordinates).all():
            raise ValueError("coordinates must be finite and match DIMENSION")
        distance_matrix = np.zeros((dimension, dimension), dtype=np.float64)
        for i in range(dimension):
            for j in range(i + 1, dimension):
                distance_matrix[i, j] = distance_matrix[j, i] = _distance(
                    coordinates[i], coordinates[j], weight_type
                )

    if not np.isfinite(distance_matrix).all() or (distance_matrix < 0).any():
        raise ValueError("edge weights must be finite and non-negative")
    return TSPInstance(name, node_ids, coordinates, distance_matrix, weight_type)
