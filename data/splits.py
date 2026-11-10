import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import tempfile
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


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


def prior_pairs(folder):
    blocked, files = set(), []
    for name in ['train.json', 'val.json', 'test.json']:
        path = folder / name
        if not path.exists():
            raise ValueError(f'Expected previous split missing: {path}')
        tasks = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(tasks, list):
            raise ValueError(f'{path} must contain a task list.')
        blocked.update((t['start'], t['target']) for t in tasks)
        files.append({'path': str(path), 'task_count': len(tasks), 'sha256': sha256(path)})
    return blocked, files


def sample_tasks(ids, matrix, count, blocked=(), seed=SEED):
    """Sample unique reachable pairs, optionally excluding previously used pairs."""
    index = {a: i for i, a in enumerate(ids)}
    reachable = sum(c not in '0_' for row in matrix for c in row)
    excluded_reachable = sum(a in index and b in index and matrix[index[a]][index[b]] not in '0_'
                             for a, b in blocked)
    available = reachable - excluded_reachable
    if count < 1 or count > available:
        raise ValueError(f'Requested {count} tasks, but only {available} eligible pairs exist.')
    rng, seen, tasks = random.Random(seed), set(), []
    while len(tasks) < count:
        i, j = rng.randrange(len(ids)), rng.randrange(len(ids))
        distance = matrix[i][j]
        pair = (ids[i], ids[j])
        if distance in '0_' or pair in blocked or pair in seen:
            continue
        seen.add(pair)
        tasks.append({'start': pair[0], 'target': pair[1], 'shortest_distance': int(distance)})
    return tasks, {'reachable_pairs': reachable, 'excluded_reachable_pairs': excluded_reachable,
                   'eligible_pairs': available}


def observed_pairs(paths, ids):
    """Recover known pairs from prior route/probe reports, without loading a model."""
    id_set = set(ids)
    display = {}
    for ident in ids:
        title = unquote(ident).replace('_', ' ')
        display.setdefault(title, set()).add(ident)

    def resolve(value):
        if value in id_set:
            return value
        matches = display.get(value, set())
        if len(matches) > 1:
            raise ValueError(f'Ambiguous article title in a debug report: {value}')
        return next(iter(matches), None)

    blocked, used = set(), []
    for path in paths:
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding='utf-8'))
        pairs = set()
        for episode in report.get('episodes', []):
            target = episode.get('target')
            for current in episode.get('path', [])[:-1]:
                if current in id_set and target in id_set and current != target:
                    pairs.add((current, target))
        for case in report.get('cases', []):
            current = resolve(case.get('current_article'))
            target = resolve(case.get('expected_title'))
            if current and target and current != target:
                pairs.add((current, target))
        blocked.update(pairs)
        used.append({'path': str(path), 'recovered_pair_count': len(pairs), 'sha256': sha256(path)})
    return blocked, used


def split_tasks(tasks, pilot_size, seed=SEED):
    """Create disjoint 80/10/10 splits and training/validation pilot subsets."""
    if len(tasks) < 10 or len(tasks) % 10:
        raise ValueError('Task count must be a positive multiple of 10, at least 10.')
    train_end = len(tasks) * 8 // 10
    if not 1 <= pilot_size <= train_end:
        raise ValueError(f'Pilot size must be between 1 and {train_end}.')
    tasks = list(tasks)
    random.Random(seed).shuffle(tasks)
    val_end = train_end + len(tasks) // 10
    splits = {
        "train": tasks[:train_end],
        "val": tasks[train_end:val_end],
        "test": tasks[val_end:],
    }
    splits['train_pilot'] = random.Random(seed).sample(splits['train'], pilot_size)
    splits['val_pilot'] = random.Random(seed).sample(splits['val'], min(100, len(splits['val'])))
    return splits


def pair_set(tasks):
    return {(t['start'], t['target']) for t in tasks}


def verify_splits(splits, blocked):
    sets = {k: pair_set(v) for k, v in splits.items()}
    for name in ['train', 'val', 'test']:
        if len(sets[name]) != len(splits[name]):
            raise ValueError(f'Duplicate pairs in {name}.')
        if sets[name] & blocked:
            raise ValueError(f'Previously used pairs leaked into {name}.')
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if sets[a] & sets[b]:
            raise ValueError(f'Overlap between {a} and {b}.')
    if not sets['train_pilot'] <= sets['train'] or not sets['val_pilot'] <= sets['val']:
        raise ValueError('A pilot set is not a subset of its full split.')
    if any(len(sets[k]) != len(v) for k, v in splits.items()):
        raise ValueError('A pilot split contains duplicate pairs.')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_experiment(output, splits, metadata):
    if output.exists():
        raise ValueError(f'Output already exists; nothing replaced: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    names = {'train': 'train.json', 'val': 'val.json', 'test': 'test.json',
             'train_pilot': f"train_pilot_{len(splits['train_pilot'])}.json",
             'val_pilot': f"val_pilot_{len(splits['val_pilot'])}.json"}
    with tempfile.TemporaryDirectory(prefix='.creating_splits_', dir=output.parent) as temp:
        staging = Path(temp) / 'ready'
        staging.mkdir()
        descriptions = {}
        for key, tasks in splits.items():
            path = staging / names[key]
            path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
            descriptions[key] = {'file': names[key], 'count': len(tasks), 'sha256': sha256(path),
                                 'shortest_distance_histogram': dict(sorted(Counter(t['shortest_distance'] for t in tasks).items()))}
        manifest = {**metadata, 'splits': descriptions,
                    'checks': {'full_splits_pair_disjoint': True,
                               'previous_pairs_excluded': True,
                               'pilot_train_is_training_subset': True,
                               'pilot_val_is_validation_subset': True},
                    'split_policy': 'Unique reachable ordered start-target pairs, uniform rejection sampling; 80/10/10 split.',
                    'pilot_policy': 'Pilot subsets intentionally overlap ONLY their own full train/validation split.'}
        (staging/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        if output.exists():
            raise ValueError(f'Output appeared during preparation; nothing replaced: {output}')
        staging.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Create reproducible Wiki Game experiment splits.")
    parser.add_argument('--graph-folder', type=Path, default=ROOT/'data/raw/graph')
    parser.add_argument('--article-order', type=Path, default=ROOT/'data/processed/article_order.json')
    parser.add_argument('--previous-splits', type=Path, nargs='+', default=[ROOT/'data/processed'],
                        help='One or more folders whose train/val/test pairs must be excluded.')
    parser.add_argument('--num-tasks', type=int, default=5000)
    parser.add_argument('--pilot-size', type=int, default=800)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or ROOT/f'data/processed/experiment_{args.num_tasks}'
    try:
        if output.exists():
            raise ValueError(f'Output already exists; nothing replaced: {output}')
        if args.num_tasks < 10 or args.num_tasks % 10:
            raise ValueError('--num-tasks must be a multiple of 10, at least 10.')
        if not 1 <= args.pilot_size <= args.num_tasks * 8 // 10:
            raise ValueError('--pilot-size must fit inside the training split.')
        ids, matrix = load_data(args.graph_folder, args.article_order)
        old_pairs, old_files = set(), []
        for folder in args.previous_splits:
            pairs, files = prior_pairs(folder)
            old_pairs |= pairs
            old_files.extend(files)
        results = ROOT / 'evaluation/results'
        debug_pairs, debug_files = observed_pairs(
            [results/name for name in ['baseline.json', 'baseline_colon_4096.json',
                                      'policy_audit.json', 'policy_audit_colon.json',
                                      'policy_decoding_probe.json', 'policy_decoding_probe_colon.json']], ids)
        blocked = old_pairs | debug_pairs
        tasks, eligibility = sample_tasks(ids, matrix, args.num_tasks, blocked)
        splits = split_tasks(tasks, args.pilot_size)
        verify_splits(splits, blocked)
        metadata = {'seed': SEED, 'num_tasks': args.num_tasks, 'article_count': len(ids),
                    'excluded_previous_pair_count': len(old_pairs),
                    'excluded_total_pair_count': len(blocked),
                    'previous_split_files': old_files, 'debug_report_files': debug_files,
                    'graph_folder': str(args.graph_folder), 'article_order': str(args.article_order),
                    'article_order_sha256': sha256(args.article_order),
                    'distance_matrix_sha256': sha256(args.graph_folder/'shortest-path-distance-matrix.txt'),
                    **eligibility}
        manifest = write_experiment(output, splits, metadata)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        parser.exit(1, f'Could not create experiment splits: {error}\n')
    print(json.dumps({'output': str(output), 'seed': SEED,
                      'excluded_previous_pair_count': len(old_pairs),
                      'excluded_total_pair_count': len(blocked),
                      'counts': {k: len(v) for k, v in splits.items()},
                      'checks': manifest['checks']}, indent=2))
    print(f'\nCreated: {output / "manifest.json"}')


if __name__ == '__main__':
    main()
