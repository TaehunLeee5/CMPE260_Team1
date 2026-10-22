import json

import pytest

from data import splits


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

    tasks = splits.sample_tasks(article_ids, matrix, 45)

    assert len(tasks) == 45
    assert len({(task["start"], task["target"]) for task in tasks}) == 45
    for task in tasks:
        start = article_ids.index(task["start"])
        target = article_ids.index(task["target"])
        assert target > start
        assert task["shortest_distance"] == target - start
    assert {"start": "Article0", "target": "Article9", "shortest_distance": 9} in tasks
    assert splits.sample_tasks(article_ids, matrix, 45) == tasks
    with pytest.raises(ValueError):
        splits.sample_tasks(article_ids, matrix, 46)


def test_split_tasks():
    """Create repeatable 80/10/10 splits with each task appearing once."""
    tasks = [
        {"start": f"A{i}", "target": "Goal", "shortest_distance": 1}
        for i in range(10)
    ]

    task_splits = splits.split_tasks(tasks)

    assert len(task_splits["train"]) == 8
    assert len(task_splits["val"]) == 1
    assert len(task_splits["test"]) == 1
    all_tasks = task_splits["train"] + task_splits["val"] + task_splits["test"]
    assert sorted(task["start"] for task in all_tasks) == [f"A{i}" for i in range(10)]
    assert splits.split_tasks(tasks) == task_splits


def test_save_splits(tmp_path):
    """Save train, validation, and test tasks as JSON files."""
    task = {"start": "A", "target": "C", "shortest_distance": 2}
    task_splits = {"train": [task], "val": [], "test": []}
    output = tmp_path / "splits"

    splits.save_splits(task_splits, output)

    for name, tasks in task_splits.items():
        assert json.loads((output / f"{name}.json").read_text(encoding="utf-8")) == tasks
