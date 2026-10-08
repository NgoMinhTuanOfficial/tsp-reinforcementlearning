from pathlib import Path

import numpy as np
import pytest

from tsp_active_search import load_tsplib


ROOT = Path(__file__).resolve().parents[1]


def test_load_coordinate_instance_and_validate_tour():
    instance = load_tsplib(ROOT / "data" / "sample10.tsp")
    assert instance.dimension == 10
    assert instance.edge_weight_type == "EUC_2D"
    assert instance.distance_matrix.shape == (10, 10)
    assert instance.tour_length(range(10)) > 0


def test_rejects_invalid_tour():
    instance = load_tsplib(ROOT / "data" / "sample10.tsp")
    with pytest.raises(ValueError, match="permutation"):
        instance.tour_length([0] * instance.dimension)


def test_explicit_upper_row_instance(tmp_path):
    path = tmp_path / "explicit.tsp"
    path.write_text(
        "NAME: explicit\nTYPE: TSP\nDIMENSION: 4\n"
        "EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: UPPER_ROW\n"
        "EDGE_WEIGHT_SECTION\n2 3 4 5 6 7\nEOF\n",
        encoding="utf-8",
    )
    instance = load_tsplib(path)
    expected = np.array(
        [[0, 2, 3, 4], [2, 0, 5, 6], [3, 5, 0, 7], [4, 6, 7, 0]],
        dtype=float,
    )
    np.testing.assert_array_equal(instance.distance_matrix, expected)
