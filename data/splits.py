import argparse
import json
import random
from pathlib import Path

def read_lines(path):
    """Read dataset lines without blank lines or comment headers."""
    with path.open(encoding="utf-8-sig") as file:
        return [line.strip() for line in file if line.strip() and not line.lstrip().startswith("#")]

def load_data(graph_folder, article_order_path):
    """Load article IDs and check their alignment with the distance matrix."""
    with article_order_path.open(encoding="utf-8") as file:
        article_ids = json.load(file)
    if not article_ids or article_ids != read_lines(graph_folder / "articles.tsv"):
        raise ValueError("The article order must match articles.tsv in the graph folder.")
    if len(article_ids) != len(set(article_ids)):
        raise ValueError("The article order contains duplicate IDs.")

    matrix = read_lines(graph_folder / "shortest-path-distance-matrix.txt")
    size = len(article_ids)
    if len(matrix) != size:
        raise ValueError("The distance matrix has the wrong number of rows.")
    for index, row in enumerate(matrix):
        if len(row) != size or set(row) - set("0123456789_") or row[index] != "0":
            raise ValueError(f"Invalid distance matrix row: {index}")
    return article_ids, matrix

def sample_tasks(article_ids, matrix, count):
    """Sample unique reachable pairs uniformly with seed 42."""
    available = sum(distance not in "0_" for row in matrix for distance in row)
    if not 1 <= count <= available:
        raise ValueError(f"Choose between 1 and {available} tasks.")

    random_generator = random.Random(42)
    chosen = set()
    tasks = []
    while len(tasks) < count:
        start = random_generator.randrange(len(article_ids))
        target = random_generator.randrange(len(article_ids))
        distance = matrix[start][target]
        if distance in "0_" or (start, target) in chosen:
            continue
        chosen.add((start, target))
        tasks.append({
            "start": article_ids[start],
            "target": article_ids[target],
            "shortest_distance": int(distance),
        })
    return tasks

def split_tasks(tasks):
    """Shuffle with seed 42 and split unique tasks into 80/10/10 sets."""
    tasks = tasks.copy()
    random.Random(42).shuffle(tasks)
    train_end = len(tasks) * 80 // 100
    val_end = train_end + len(tasks) * 10 // 100
    return {
        "train": tasks[:train_end],
        "val": tasks[train_end:val_end],
        "test": tasks[val_end:],
    }

def save_splits(splits, output_folder):
    """Save each split as a JSON list of start, target, and shortest distance."""
    output_folder.mkdir(parents=True, exist_ok=True)
    for name, tasks in splits.items():
        with (output_folder / f"{name}.json").open("w", encoding="utf-8") as file:
            json.dump(tasks, file, ensure_ascii=False, indent=2)

def main():
    """Sample navigation tasks and save train, validation, and test splits."""
    data_folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Sample Wikispeedia navigation tasks.")
    parser.add_argument("--graph-folder", type=Path, required=True)
    parser.add_argument("--num-tasks", type=int, required=True, help="Number of unique tasks to sample")
    args = parser.parse_args()
    article_ids, matrix = load_data(args.graph_folder, data_folder / "processed" / "article_order.json")
    tasks = sample_tasks(article_ids, matrix, args.num_tasks)
    splits = split_tasks(tasks)
    output_folder = data_folder / "processed"
    save_splits(splits, output_folder)
    for name, records in splits.items():
        print(f"{name}: {len(records):,} tasks")
    print(f"Saved JSON files to: {output_folder}")

if __name__ == "__main__":
    main()
