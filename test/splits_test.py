import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from data import splits


@pytest.fixture
def graph():
    ids = [f"Article{i}" for i in range(8)]
    matrix = ["".join("0" if i == j else "1" for j in range(8)) for i in range(8)]
    return ids, matrix


def test_excluded_pairs_and_capacity(graph):
    ids, matrix = graph
    blocked = {(ids[0], ids[1]), (ids[1], ids[0])}
    tasks, eligibility = splits.sample_tasks(ids, matrix, 54, blocked)

    assert len(tasks) == len(splits.pair_set(tasks)) == 54
    assert not splits.pair_set(tasks) & blocked
    assert all(t["start"] != t["target"] and t["shortest_distance"] == 1 for t in tasks)
    assert eligibility == {
        "reachable_pairs": 56, "excluded_reachable_pairs": 2, "eligible_pairs": 54,
    }
    assert splits.sample_tasks(ids, matrix, 54, blocked)[0] == tasks
    with pytest.raises(ValueError, match="54 eligible pairs"):
        splits.sample_tasks(ids, matrix, 55, blocked)


def test_pilot_subsets_and_overlap_detection(graph):
    ids, matrix = graph
    tasks, _ = splits.sample_tasks(ids, matrix, 40)
    partition = splits.split_tasks(tasks, pilot_size=12)
    splits.verify_splits(partition, blocked=set())

    assert {name: len(rows) for name, rows in partition.items()} == {
        "train": 32, "val": 4, "test": 4, "train_pilot": 12, "val_pilot": 4,
    }
    assert splits.pair_set(partition["train_pilot"]) <= splits.pair_set(partition["train"])
    assert splits.pair_set(partition["val_pilot"]) <= splits.pair_set(partition["val"])
    partition["test"][0] = partition["train"][0]
    with pytest.raises(ValueError, match="Overlap"):
        splits.verify_splits(partition, blocked=set())


def test_cli_manifest_and_no_overwrite(tmp_path, graph):
    ids, matrix = graph
    graph_folder = tmp_path / "graph"
    graph_folder.mkdir()
    (graph_folder / "articles.tsv").write_text("\n".join(ids), encoding="utf-8")
    (graph_folder / "shortest-path-distance-matrix.txt").write_text(
        "\n".join(matrix), encoding="utf-8",
    )
    order = tmp_path / "article_order.json"
    order.write_text(json.dumps(ids), encoding="utf-8")
    previous = tmp_path / "previous"
    previous.mkdir()
    for name in ["train", "val", "test"]:
        rows = [{"start": ids[0], "target": ids[1]}] if name == "train" else []
        (previous / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")
    output = tmp_path / "experiment"
    args = [
        sys.executable, "-m", "data.splits",
        "--graph-folder", str(graph_folder), "--article-order", str(order),
        "--previous-splits", str(previous), "--output", str(output),
        "--num-tasks", "40", "--pilot-size", "12",
    ]
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(args, cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["excluded_previous_pair_count"] == 1
    assert all(manifest["checks"].values())
    all_pairs = set()
    for name, entry in manifest["splits"].items():
        path = output / entry["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        rows = json.loads(path.read_text(encoding="utf-8"))
        assert len(rows) == entry["count"]
        if name in {"train", "val", "test"}:
            all_pairs.update(splits.pair_set(rows))
    assert len(all_pairs) == 40
    assert (ids[0], ids[1]) not in all_pairs

    before = {path.name: path.read_bytes() for path in output.iterdir()}
    retry = subprocess.run(args, cwd=root, capture_output=True, text=True)
    assert retry.returncode != 0
    assert "already exists" in retry.stderr
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before



def test_read_lines(tmp_path):
    """Read dataset lines while skipping comments and blank lines."""
    path = tmp_path / "articles.tsv"
    path.write_text("# Articles\n\nA\nB\n", encoding="utf-8")

    assert splits.read_lines(path) == ["A", "B"]


def test_load_data(tmp_path):
    """Load matching article IDs and shortest-path distances."""
    article_ids = ["A", "B", "C"]
    matrix = ["012", "_01", "__0"]
    order_path = tmp_path / "article_order.json"
    order_path.write_text(json.dumps(article_ids), encoding="utf-8")
    (tmp_path / "articles.tsv").write_text("A\nB\nC\n", encoding="utf-8")
    (tmp_path / "shortest-path-distance-matrix.txt").write_text(
        "\n".join(matrix), encoding="utf-8"
    )

    assert splits.load_data(tmp_path, order_path) == (article_ids, matrix)


def test_sample_tasks():
    """Sample valid pairs from a directed chain with a true nine-click route."""
    article_ids = [f"Article{i}" for i in range(10)]
    matrix = []
    for start in range(10):
        row = [str(target - start) if target >= start else "_" for target in range(10)]
        matrix.append("".join(row))

    tasks, _ = splits.sample_tasks(article_ids, matrix, 45)

    assert len(tasks) == 45
    assert len({(task["start"], task["target"]) for task in tasks}) == 45
    for task in tasks:
        start = article_ids.index(task["start"])
        target = article_ids.index(task["target"])
        assert target > start
        assert task["shortest_distance"] == target - start
    assert {"start": "Article0", "target": "Article9", "shortest_distance": 9} in tasks
    assert splits.sample_tasks(article_ids, matrix, 45)[0] == tasks
    with pytest.raises(ValueError):
        splits.sample_tasks(article_ids, matrix, 46)
