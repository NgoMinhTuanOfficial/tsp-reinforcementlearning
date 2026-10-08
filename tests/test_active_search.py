from pathlib import Path

from tsp_active_search import ActiveSearchConfig, active_search, load_tsplib


ROOT = Path(__file__).resolve().parents[1]


def test_active_search_returns_valid_tour_and_history():
    instance = load_tsplib(ROOT / "data" / "sample10.tsp")
    result = active_search(
        instance,
        ActiveSearchConfig(
            iterations=2,
            batch_size=3,
            hidden_dim=16,
            seed=11,
            device="cpu",
        ),
    )
    assert len(result.best_tour) == instance.dimension
    assert sorted(result.best_tour) == list(range(instance.dimension))
    assert result.best_length == instance.tour_length(result.best_tour)
    assert len(result.history) == 2
    assert result.history[-1] <= result.history[0]
