import json
import math
import random
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from environment.environment import WikiGameEnv
from environment.episode_runner import EpisodeRunner


class TargetPolicy:
    """Deterministic test policy: take the target, first link, or forced Back."""

    def __init__(self):
        self.deterministic_calls = []

    def choose_action(self, observation, deterministic=False):
        self.deterministic_calls.append(deterministic)
        for action in observation["actions"]:
            if action["title"] == observation["target_title"]:
                return self.follow(action)
        if observation["actions"]:
            return self.follow(observation["actions"][0])
        return {
            "action": observation["back_action_id"],
            "tool_call": {"name": "go_back", "arguments": {}}, "log_prob": 0.0,
        }

    @staticmethod
    def follow(action):
        return {
            "action": action["id"],
            "tool_call": {"name": "follow_link", "arguments": {"title": action["title"]}},
            "log_prob": 0.0,
        }


@pytest.fixture
def runner(tmp_path):
    """Exercise the runner on a small graph without model downloads or a GPU."""
    articles = {
        "Europe": {
            "title": "Europe",
            "text": "Europe includes Germany and France.",
            "outgoing": ["Germany", "France"],
        },
        "Germany": {
            "title": "Germany",
            "text": "Germany borders France.",
            "outgoing": ["France"],
        },
        "France": {
            "title": "France",
            "text": "France is a country in Europe.",
            "outgoing": ["Europe"],
        },
        "Loop": {"title": "Loop", "text": "This article links to itself.", "outgoing": ["Loop", "Germany"]},
        "Dead_end_start": {
            "title": "Dead end start",
            "text": "This article only links to a dead end.",
            "outgoing": ["Dead_end"],
        },
        "Dead_end": {"title": "Dead end", "text": "No outgoing links.", "outgoing": []},
    }
    path = tmp_path / "articles.json"
    path.write_text(json.dumps(articles), encoding="utf-8")
    env = WikiGameEnv(path, max_clicks=2)
    yield EpisodeRunner(env, TargetPolicy())
    env.close()


@pytest.mark.parametrize("deterministic", [False, True])
def test_run_episode(runner, deterministic):
    """Record the policy's decisions and forward its sampling-mode argument."""
    episode = runner.run_episode("Europe", "France", deterministic=deterministic)

    assert episode["start"] == "Europe"
    assert episode["target"] == "France"
    assert episode["success"] and episode["reason"] == "success"
    assert episode["terminated"] and not episode["truncated"]
    assert 1 <= episode["clicks"] <= 2
    assert episode["clicks"] == len(episode["steps"])
    assert len(episode["path"]) == episode["clicks"] + 1
    assert episode["path"][0] == "Europe"
    assert episode["path"][-1] == "France"

    for index, step in enumerate(episode["steps"]):
        current = runner.env.articles[episode["path"][index]]
        observation = step["observation"]
        assert observation["current_title"] == current["title"]
        assert observation["target_title"] == "France"
        assert observation["remaining_clicks"] == runner.env.max_clicks - index
        assert current["outgoing"][step["action"]] == episode["path"][index + 1]
        destination = runner.env.articles[current["outgoing"][step["action"]]]["title"]
        assert step["tool_call"] == {"name": "follow_link", "arguments": {"title": destination}}
        assert isinstance(step["log_prob"], float)
        assert math.isfinite(step["log_prob"]) and step["log_prob"] <= 0
        assert step["log_prob"] == 0.0
    assert runner.policy.deterministic_calls == [deterministic] * episode["clicks"]

    rewards = [step["reward"] for step in episode["steps"]]
    expected_reward = 1.05 if episode["clicks"] == 1 else 1.0
    assert all(reward == 0 for reward in rewards[:-1])
    assert rewards[-1] == pytest.approx(expected_reward)
    assert episode["episode_return"] == pytest.approx(sum(rewards))
    assert json.loads(json.dumps(episode)) == episode
    print(f"\nPath: {episode['path']}, reward: {episode['episode_return']}")


@pytest.mark.parametrize("start, reason, clicks", [
    ("Loop", "click_budget", 2),
    ("Dead_end_start", "click_budget", 2),
])
def test_failed_episode(runner, start, reason, clicks):
    """Stop at the budget, allowing recovery from a page without links."""
    episode = runner.run_episode(start, "France")

    assert not episode["success"]
    assert episode["reason"] == reason
    assert episode["terminated"] and not episode["truncated"]
    assert episode["clicks"] == clicks
    assert len(episode["steps"]) == clicks
    assert episode["episode_return"] == 0
    assert all(step["reward"] == 0 for step in episode["steps"])
    if start == "Dead_end_start":
        assert episode["path"] == ["Dead_end_start", "Dead_end", "Dead_end_start"]
        assert episode["navigation_stack"] == ["Dead_end_start"]
        assert episode["steps"][1]["action"] == runner.env.back_action
        assert episode["steps"][1]["tool_call"] == {"name": "go_back", "arguments": {}}
        assert episode["steps"][1]["observation"]["back_available"]
        assert episode["steps"][1]["observation"]["history"] == ["Dead end start", "Dead end"]


def test_runner_reuse(runner):
    """Reset between games while preserving the data from earlier episodes."""
    first = runner.run_episode("Loop", "France")
    original = deepcopy(first)
    second = runner.run_episode("Germany", "France", deterministic=True)

    assert first == original
    assert second["success"] and second["reason"] == "success"
    assert second["path"] == ["Germany", "France"]
    assert second["clicks"] == 1
    assert second["steps"][0]["observation"]["remaining_clicks"] == runner.env.max_clicks
    assert second["episode_return"] == pytest.approx(1.05)


def test_evaluation_keeps_compact_tool_trace(runner):
    from evaluation.evaluate import evaluate

    report = evaluate(runner, [{"start": "Europe", "target": "France", "shortest_distance": 1}])
    episode = report["episodes"][0]
    assert report["summary"]["successes"] == 1
    assert episode["tool_calls"] == [{"name": "follow_link", "arguments": {"title": "France"}}]
    assert episode["navigation_stack"] == ["Europe", "France"]
    assert episode["action_log_probs"] == [0.0]
    assert report["settings"]["deterministic"] is True
    assert "groups" not in report
    assert "steps" not in episode  # Evaluation files omit bulky training inputs.
    assert json.loads(json.dumps(report)) == report


class SamplePolicy(TargetPolicy):
    """Use seeded draws on the reachable graph; deliberately stay in Loop."""

    def choose_action(self, observation, deterministic=False):
        self.deterministic_calls.append(deterministic)
        if observation["current_title"] == "Loop":
            return self.follow(observation["actions"][0])
        options = [self.follow(action) for action in observation["actions"]]
        if observation["back_available"]:
            options.append({
                "action": observation["back_action_id"],
                "tool_call": {"name": "go_back", "arguments": {}}, "log_prob": 0.0,
            })
        choice = options[int(random.random() * len(options))]
        choice["log_prob"] = -math.log(len(options))
        return choice


def test_sampled_groups_are_reproducible_and_seeded_once(runner, monkeypatch):
    from evaluation import evaluate as evaluation

    calls = []
    seed = evaluation.set_seed

    def record_seed(value):
        calls.append(value)
        seed(value)

    monkeypatch.setattr(evaluation, "set_seed", record_seed)
    runner.policy = SamplePolicy()
    tasks = [
        {"start": "Europe", "target": "France", "shortest_distance": 1},
        {"start": "Loop", "target": "France", "shortest_distance": 2},
    ]
    result = evaluation.evaluate(runner, tasks, deterministic=False, rollouts=4, seed=42)
    assert calls == [42]
    assert len(result["episodes"]) == 8
    assert all(mode is False for mode in runner.policy.deterministic_calls)
    assert result["settings"] == {
        "max_clicks": 2, "deterministic": False, "rollouts_per_task": 4, "rollout_seed": 42,
    }
    varied, failed = result["groups"]
    assert varied["returns"] == pytest.approx([1.05, 1.0, 0.0, 1.05])
    assert varied["successes"] == 3
    assert varied["has_reward_variation"] and varied["return_std"] > 0
    assert varied["unique_routes"] == 3
    assert failed["returns"] == [0.0] * 4
    assert not failed["has_reward_variation"] and failed["return_std"] == 0
    assert result["group_summary"]["mixed_success_groups"] == 1
    assert result["group_summary"]["groups_with_reward_variation"] == 1
    assert result["group_summary"]["zero_return_groups"] == 1
    assert result["route_audit"]["all_recorded_transitions_and_returns_valid"]
    assert [(e["task_index"], e["rollout_index"]) for e in result["episodes"]] == [
        (task, rollout) for task in [1, 2] for rollout in [1, 2, 3, 4]
    ]
    repeated = evaluation.evaluate(runner, tasks, deterministic=False, rollouts=4, seed=42)
    assert repeated == result


def test_greedy_evaluation_does_not_reseed(runner, monkeypatch):
    from evaluation import evaluate as evaluation

    def unexpected_seed(*args, **kwargs):
        raise AssertionError("Greedy evaluation should not reset sampling RNGs.")

    monkeypatch.setattr(evaluation, "set_seed", unexpected_seed)
    evaluation.evaluate(runner, [{"start": "Europe", "target": "France", "shortest_distance": 1}])


@pytest.mark.parametrize("options", [
    {"rollouts": 0}, {"rollouts": 4}, {"seed": -1}, {"seed": 2**32},
])
def test_invalid_rollout_settings_rejected(runner, options):
    from evaluation.evaluate import evaluate

    with pytest.raises(ValueError):
        evaluate(runner, [{"start": "Europe", "target": "France", "shortest_distance": 1}], **options)


def test_duplicate_task_groups_rejected(runner):
    from evaluation.evaluate import evaluate

    task = {"start": "Europe", "target": "France", "shortest_distance": 1}
    with pytest.raises(ValueError, match="unique start-target"):
        evaluate(runner, [task, task], deterministic=False, rollouts=4)


def test_route_diversity_is_separate_from_reward_variation():
    from evaluation.evaluate import summarize_groups

    rows = [{
        "task_index": 1, "start": "A", "target": "T", "shortest_distance": 2,
        "episode_return": 0.0, "success": False, "path": path,
        "tool_calls": [{"name": "follow_link", "arguments": {"title": link}}],
    } for path, link in [(["A", "B"], "B"), (["A", "C"], "C")]]
    groups, summary = summarize_groups(rows)
    assert groups[0]["unique_routes"] == groups[0]["unique_tool_sequences"] == 2
    assert groups[0]["return_std"] == 0
    assert summary["groups_with_reward_variation"] == 0
    assert summary["zero_return_groups"] == 1


def test_successful_groups_can_vary_by_efficiency_bonus():
    from evaluation.evaluate import summarize_groups

    rows = [{
        "task_index": 1, "start": "A", "target": "T", "shortest_distance": 1,
        "episode_return": reward, "success": True, "path": path, "tool_calls": [],
    } for reward, path in [(1.05, ["A", "T"]), (1.0, ["A", "B", "T"])]]
    groups, summary = summarize_groups(rows)
    assert groups[0]["has_reward_variation"]
    assert groups[0]["return_std"] == pytest.approx(0.025)
    assert summary["mixed_success_groups"] == 0
    assert summary["groups_with_reward_variation"] == 1


def test_cli_filters_distance_before_selecting_tasks(runner, tmp_path, monkeypatch):
    from evaluation import evaluate as evaluation

    tasks = [
        {"start": "Europe", "target": "France", "shortest_distance": 1},
        {"start": "Germany", "target": "France", "shortest_distance": 1},
        {"start": "Loop", "target": "France", "shortest_distance": 2},
    ]
    task_path = tmp_path / "tasks.json"
    task_path.write_text(json.dumps(tasks), encoding="utf-8")
    output = tmp_path / "results" / "sampled.json"
    policy = runner.policy
    policy.model = SimpleNamespace(name_or_path="test-policy")
    policy.max_input_tokens = 4096
    policy.system_prompt = "Test prompt"
    policy.observation_template = "Test observation"
    monkeypatch.setattr(evaluation, "WikiGameEnv", lambda **kwargs: runner.env)
    monkeypatch.setattr(evaluation, "WikiGamePolicy", lambda **kwargs: policy)
    monkeypatch.setattr(sys, "argv", [
        "evaluate", "--tasks", str(task_path), "--output", str(output),
        "--distance", "2", "--limit", "1", "--seed", "1", "--sample",
        "--rollouts", "4", "--max-clicks", "2",
    ])
    evaluation.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["settings"]["distance_filter"] == 2
    assert report["group_summary"]["num_tasks"] == 1
    assert report["summary"]["num_episodes"] == 4
    assert all(e["start"] == "Loop" and e["shortest_distance"] == 2 for e in report["episodes"])
    assert report["route_audit"]["all_recorded_transitions_and_returns_valid"]


@pytest.mark.parametrize("distance, message", [(0, "must be positive"), (3, "No tasks have")])
def test_cli_rejects_unusable_distance_before_loading_model(tmp_path, monkeypatch, capsys,
                                                          distance, message):
    from evaluation import evaluate as evaluation

    task_path = tmp_path / "tasks.json"
    task_path.write_text(json.dumps([
        {"start": "Europe", "target": "France", "shortest_distance": 1},
    ]), encoding="utf-8")

    def unexpected_model(**kwargs):
        raise AssertionError("Invalid task selection should fail before model loading.")

    monkeypatch.setattr(evaluation, "WikiGamePolicy", unexpected_model)
    monkeypatch.setattr(sys, "argv", [
        "evaluate", "--tasks", str(task_path), "--output", str(tmp_path / "out.json"),
        "--distance", str(distance),
    ])
    with pytest.raises(SystemExit) as error:
        evaluation.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("config_only", [False, True])
def test_cli_rejects_missing_adapter_weights_before_loading_model(tmp_path, monkeypatch, capsys,
                                                                config_only):
    from evaluation import evaluate as evaluation

    adapter = tmp_path / "adapter"
    if config_only:
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    def unexpected_model(**kwargs):
        raise AssertionError("An invalid adapter path should fail before model loading.")

    monkeypatch.setattr(evaluation, "WikiGamePolicy", unexpected_model)
    monkeypatch.setattr(sys, "argv", [
        "evaluate", "--tasks", str(tmp_path / "train.json"),
        "--adapter", str(adapter), "--output", str(tmp_path / "out.json"),
    ])
    with pytest.raises(SystemExit) as error:
        evaluation.main()
    assert error.value.code == 2
    assert "--adapter must contain" in capsys.readouterr().err
