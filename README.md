# Policy Optimization for an LLM Wiki Game Agent

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Run all commands from the project root.

## Setup

```bash
uv sync
```

## Preprocess the dataset

Download and extract the graph and plaintext archives from
[Wikispeedia](https://snap.stanford.edu/data/wikispeedia.html).

- Put `articles.tsv`, `links.tsv`, and `shortest-path-distance-matrix.txt` in `data/raw/graph/`.
- Put the article `.txt` files in `data/raw/plaintext/`.

```bash
uv run data/preprocess.py --graph-folder "data/raw/graph" --text-folder "data/raw/plaintext"
```

Replace the input paths if your folders are elsewhere. The script creates
`articles.json` and `article_order.json` in `data/processed/`.

## Create task splits

After preprocessing, sample unique, reachable start–target pairs from the distance matrix:

```bash
uv run data/splits.py --graph-folder "data/raw/graph" --num-tasks 1000
```

Replace `1000` with the number of tasks you want to sample. The script saves
`train.json`, `val.json`, and `test.json` in `data/processed/` using an 80/10/10 split.
Each task contains `start`, `target`, and `shortest_distance`.

## Run tests

```bash
uv run pytest test
```
