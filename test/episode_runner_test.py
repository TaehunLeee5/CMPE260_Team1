import json
import math
from copy import deepcopy

import pytest
import torch

from environment.environment import WikiGameEnv
from environment.episode_runner import EpisodeRunner
from policy.llm_policy import WikiGamePolicy


@pytest.fixture(scope="module")
def wiki_policy():
    """Load the real Qwen model once for this test file."""
    return WikiGamePolicy()


@pytest.fixture
def runner(tmp_path, wiki_policy):
    """Use the real policy with a small graph containing success and failure routes."""
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
    yield EpisodeRunner(env, wiki_policy)
    env.close()


@pytest.mark.parametrize("deterministic", [False, True])
def test_run_episode(runner, deterministic):
    """Record a complete successful episode with matching actions and model scores."""
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
        assert isinstance(step["log_prob"], float)
        assert math.isfinite(step["log_prob"]) and step["log_prob"] <= 0
        with torch.no_grad():
            recomputed = runner.policy.get_action_log_prob(observation, step["action"])
        assert recomputed.item() == pytest.approx(step["log_prob"], abs=1e-4)

    rewards = [step["reward"] for step in episode["steps"]]
    expected_reward = 1.05 if episode["clicks"] == 1 else 1.0
    assert all(reward == 0 for reward in rewards[:-1])
    assert rewards[-1] == pytest.approx(expected_reward)
    assert episode["episode_return"] == pytest.approx(sum(rewards))
    assert json.loads(json.dumps(episode)) == episode
    print(f"\nPath: {episode['path']}, reward: {episode['episode_return']}")


@pytest.mark.parametrize("start, reason, clicks", [
    ("Loop", "click_budget", 2),
    ("Dead_end_start", "dead_end", 1),
])
def test_failed_episode(runner, start, reason, clicks):
    """Stop at the click budget or a dead end and keep all failed-route rewards zero."""
    episode = runner.run_episode(start, "France")

    assert not episode["success"]
    assert episode["reason"] == reason
    assert episode["terminated"] and not episode["truncated"]
    assert episode["clicks"] == clicks
    assert len(episode["steps"]) == clicks
    assert episode["episode_return"] == 0
    assert all(step["reward"] == 0 for step in episode["steps"])


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
