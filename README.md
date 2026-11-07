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

Create new task splits, excluding pairs from the existing splits in
`data/processed/`:

```bash
uv run python -m data.splits --num-tasks 5000 --pilot-size 800
```

The script saves train/validation/test splits (80/10/10), pilot subsets, and a
manifest to `data/processed/experiment_5000/`. Existing directories are not overwritten.

## Run tests

```bash
uv run pytest test
```
