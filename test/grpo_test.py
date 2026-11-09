import math

import pytest
import torch

import json
import random
import sys
from collections import Counter

from training import grpo
from training.grpo import (clipped_token_loss, distance_list, group_advantages,
                           sample_batch_tasks)


def test_group_advantages_use_population_std_and_handle_constant_rewards():
    expected = 1 / (math.sqrt(2 / 3) + 1e-4)
    assert group_advantages([0, 1, 2]) == pytest.approx([-expected, 0, expected])
    assert group_advantages([0, 0, 0, 0]) == [0.0] * 4
    assert group_advantages([1.09] * 4) == [0.0] * 4


@pytest.mark.parametrize("returns", [[1], [float("nan"), 0], [0, float("inf")]])
def test_invalid_group_rewards_rejected(returns):
    with pytest.raises(ValueError):
        group_advantages(returns)


@pytest.mark.parametrize("ratio, advantage, expected_loss, expected_gradient", [
    (1.5, 1, -1.2, 0.0),
    (0.5, -1, 0.8, 0.0),
    (0.5, 1, -0.5, -0.5),
    (1.5, -1, 1.5, 1.5),
])
def test_clipping_direction_and_detached_behavior(ratio, advantage, expected_loss, expected_gradient):
    old = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)
    current = torch.tensor(-2.0 + math.log(ratio), dtype=torch.float64, requires_grad=True)
    loss = clipped_token_loss(current, old, advantage)
    loss.backward()
    assert loss.item() == pytest.approx(expected_loss)
    assert current.grad.item() == pytest.approx(expected_gradient)
    assert old.grad is None


def test_std_floor_shrinks_efficiency_only_groups():
    # All routes succeed; lengths differ by one click (return gap 0.005).
    returns = [1.09, 1.09, 1.085, 1.085]
    unfloored = group_advantages(returns)
    floored = group_advantages(returns, min_std=0.1)
    assert unfloored == pytest.approx([0.96, 0.96, -0.96, -0.96], abs=0.01)
    assert floored == pytest.approx([0.025, 0.025, -0.025, -0.025])


def test_std_floor_leaves_success_failure_groups_unchanged():
    returns = [1.08, 0.0, 0.0, 1.09, 0.0, 0.0, 0.0, 0.0]
    std = math.sqrt(sum((r - sum(returns) / 8) ** 2 for r in returns) / 8)
    assert std > 0.1
    expected = [(r - sum(returns) / 8) / std for r in returns]
    assert group_advantages(returns, min_std=0.1) == pytest.approx(expected)
    assert group_advantages([0.0] * 8, min_std=0.1) == [0.0] * 8
    assert group_advantages([1.09] * 8, min_std=0.1) == [0.0] * 8


@pytest.mark.parametrize("min_std", [0, -0.1, float("nan"), float("inf")])
def test_invalid_std_floor_rejected(min_std):
    with pytest.raises(ValueError):
        group_advantages([0, 1], min_std=min_std)


def test_distance_list_accepts_legacy_integer():
    assert distance_list(2) == [2]
    assert distance_list([4, 2, 3, 2]) == [2, 3, 4]
    for bad in [[], [0], [2, "3"], [2.0]]:
        with pytest.raises(ValueError):
            distance_list(bad)


def _tasks(counts):
    return [{"start": f"s{d}_{i}", "target": f"t{d}_{i}", "shortest_distance": d}
            for d, n in counts.items() for i in range(n)]


def test_single_distance_sampling_matches_original_behavior():
    tasks = _tasks({2: 50})
    assert sample_batch_tasks(tasks, [2], 4, random.Random(7)) == random.Random(7).sample(tasks, 4)


def test_multi_distance_sampling_is_uniform_over_distances():
    tasks = _tasks({2: 10, 3: 100, 4: 50})  # Pool sizes deliberately unequal.
    rng = random.Random(0)
    counts = Counter()
    for _ in range(3000):
        batch = sample_batch_tasks(tasks, [2, 3, 4], 4, rng)
        assert len(batch) == 4 and len({(t["start"], t["target"]) for t in batch}) == 4
        counts.update(t["shortest_distance"] for t in batch)
    for d in [2, 3, 4]:
        assert counts[d] / 12000 == pytest.approx(1 / 3, abs=0.02)
    assert sample_batch_tasks(tasks, [2, 3, 4], 4, random.Random(5)) == \
        sample_batch_tasks(tasks, [2, 3, 4], 4, random.Random(5))


def test_cli_rejects_distance_without_enough_tasks(tmp_path, monkeypatch, capsys):
    train, val = tmp_path / "train.json", tmp_path / "val.json"
    train.write_text(json.dumps(_tasks({2: 10, 3: 10, 4: 2})), encoding="utf-8")
    val.write_text(json.dumps([{"start": "v", "target": "w", "shortest_distance": 2}]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["grpo", "--tasks", str(train), "--val-tasks", str(val),
                                      "--distance", "2", "3", "4", "--output", str(tmp_path / "run")])
    with pytest.raises(SystemExit):
        grpo.main()
    assert "distance(s): [4]" in capsys.readouterr().err


def test_defaults_match_final_training_setup():
    assert grpo.DEFAULTS["distance"] == [2, 3, 4]
    assert grpo.DEFAULTS["rollouts"] == 8 and grpo.DEFAULTS["tasks_per_batch"] == 4
    assert grpo.DEFAULTS["min_advantage_std"] == 0.1
    assert grpo.DEFAULTS["update_epochs"] == 2
    assert grpo.DEFAULTS["val_limit"] == 100 and grpo.DEFAULTS["eval_every"] == 10
    assert "final_v2_5000" in grpo.DEFAULTS["tasks"] and "final_v2_5000" in grpo.DEFAULTS["val_tasks"]
