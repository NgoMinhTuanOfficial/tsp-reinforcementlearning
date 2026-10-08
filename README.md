# TSP Active Search

Solve Traveling Salesperson Problem instances in TSPLIB format with **per-instance Active Search**. The Pointer Network starts from random initialization for each input instance; there is no pretraining or pretrained checkpoint. Its policy is refined online using sampled tours and an exponential-moving-average reward baseline.

## Features

- Modular Python source under `src/tsp_active_search/`.
- TSPLIB `.tsp` reader for coordinate instances and explicit edge-weight matrices.
- Active Search, greedy decoding, and multi-sample selection.
- Notebook example that runs the source and plots the best tour and search history.
- CPU by default; CUDA is used automatically when available.

## Install

Python 3.10 or later is required.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install -e ".[notebook]"
```

## Run

Pass a TSPLIB instance to the command line:

```bash
python -m tsp_active_search data/sample10.tsp --iterations 100 --batch-size 32 --seed 7
```

Or use the Python API:

```python
from tsp_active_search import ActiveSearchConfig, active_search, load_tsplib

instance = load_tsplib("data/sample10.tsp")
result = active_search(
    instance,
    ActiveSearchConfig(iterations=100, batch_size=32, seed=7),
)
print(result.best_length, result.best_tour)
```

`best_tour` contains zero-based node indices in visit order. TSPLIB node labels are available as `instance.node_ids`.

## Notebook

Open `notebooks/active_search.ipynb` after installing the optional notebook dependencies. It loads the included TSPLIB-format demo instance, runs Active Search, and plots the best tour and incumbent length by iteration. To use another benchmark, change `INSTANCE_PATH` to its `.tsp` file.

## Algorithm

For an instance, initialize the policy randomly, then repeat: sample a batch of tours, score their TSPLIB tour lengths, update the policy with the REINFORCE loss and EMA baseline, and retain the best tour seen. There is no offline training phase. `iterations` is the number of online updates; each update samples `batch_size` tours.

The parser supports `EUC_2D`, `CEIL_2D`, `ATT`, `GEO`, `MAN_2D`, `MAX_2D`, and `EXPLICIT` edge weights (including the standard matrix formats). Unsupported TSPLIB edge-weight types or malformed instances raise clear errors.

## Tests

```bash
python -m pytest
```

## Scope

This is a research/education implementation. Active Search is stochastic and does not guarantee an optimal tour; increase iterations and batch size for more search at higher compute cost.
