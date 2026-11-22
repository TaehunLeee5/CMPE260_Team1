# Policy Optimization for an LLM Wiki Game Agent

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Run all commands from the project root.

```bash
uv sync
```

## 1. Preprocess the dataset

Download and extract the graph and plaintext archives from
[Wikispeedia](https://snap.stanford.edu/data/wikispeedia.html).

- Put `articles.tsv`, `links.tsv`, and `shortest-path-distance-matrix.txt` in `data/raw/graph/`.
- Put the article `.txt` files in `data/raw/plaintext/`.

```bash
uv run data/preprocess.py --graph-folder "data/raw/graph" --text-folder "data/raw/plaintext"
```

## 2. Training splits

The splits used for training are included in the repo at `data/processed/final_v2_5000/`
(4,000 train / 500 validation / 500 test pairs). Use these instead of regenerating them:
`data/splits.py` would produce different pairs on another machine.

## 3. Train GRPO

```bash
uv run python -m training.grpo --batches 100 --no-revisits --output runs/grpo
```

The defaults are the final setup. Results and checkpoints are saved in `runs/grpo/`;
`report.json` lists validation results and the best checkpoint. To continue a stopped run:

```bash
uv run python -m training.grpo --resume runs/grpo --batches 100
```

## 4. Evaluate

Base model on validation:

```bash
uv run python -m evaluation.evaluate --tasks data/processed/final_v2_5000/val.json --no-revisits --output evaluation/results/base_val.json
```

Trained GRPO model on validation:

```bash
uv run python -m evaluation.evaluate --tasks data/processed/final_v2_5000/val.json --no-revisits --adapter runs/grpo/checkpoints/batch_000070/adapter --output evaluation/results/grpo_val.json
```

Base model on test:

```bash
uv run python -m evaluation.evaluate --tasks data/processed/final_v2_5000/test.json --no-revisits --output evaluation/results/base_test.json
```

Trained GRPO model on test:

```bash
uv run python -m evaluation.evaluate --tasks data/processed/final_v2_5000/test.json --no-revisits --adapter runs/grpo/checkpoints/batch_000070/adapter --output evaluation/results/grpo_test.json
```

Replace `batch_000070` with the best checkpoint listed in `runs/grpo/report.json`.

## 5. Settings for a fair comparison

Keep these the same for every method (PPO, GRPO, GSPO):

| Setting | Value |
|---|---|
| Data | `data/processed/final_v2_5000` splits |
| Model | Qwen3-1.7B, LoRA rank 8, alpha 16, dropout 0, all linear layers |
| Environment | 20 actions, self-links removed, no-revisits rule (`--no-revisits`) |
| Prompt | Article text truncated to 4,096 tokens (`--prompt-style article_text`) |
| Reward | `progress` for training; evaluation scores with the success reward |
| Training budget | 3,200 episodes (GRPO: 100 batches × 32 episodes) |
| Validation | 200-pair subset of `val.json` (seed 42), every 10 batches; keep the best checkpoint |
| Evaluation | Greedy decoding; `test.json` only once, for the final model |

Tune these per method on validation: learning rate (GRPO: 3e-5), clip range (GRPO: 0.2;
GSPO needs a much smaller range because its ratio is per route), and method-specific
settings such as PPO's value network. GSPO should use the same group size (8 rollouts per
task) and number of update epochs (2) as GRPO.
