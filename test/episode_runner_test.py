import json
import math
from copy import deepcopy

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
                return self.follow(action["id"])
        if observation["actions"]:
            return self.follow(observation["actions"][0]["id"])
        return {
            "action": observation["back_action_id"],
            "tool_call": {"name": "go_back", "arguments": {}}, "log_prob": 0.0,
        }

    @staticmethod
    def follow(link_id):
        return {
            "action": link_id,
            "tool_call": {"name": "follow_link", "arguments": {"link_id": link_id}},
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
        "Loop": {"title": "Loop", "text": "This article links to itself.", "outgoing": ["Loop"]},
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
        assert step["tool_call"] == {"name": "follow_link", "arguments": {"link_id": step["action"]}}
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
    assert episode["tool_calls"] == [{"name": "follow_link", "arguments": {"link_id": 1}}]
    assert episode["navigation_stack"] == ["Europe", "France"]
    assert "steps" not in episode  # Evaluation files omit bulky training inputs.
    assert json.loads(json.dumps(report)) == report
