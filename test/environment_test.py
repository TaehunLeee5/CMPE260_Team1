import json

import gymnasium as gym
import pytest

from environment.environment import WikiGameEnv


@pytest.fixture
def env(tmp_path):
    """Create a small environment with a goal, a loop, and a dead end."""
    articles = {
        "Start": {
            "title": "Start",
            "text": "Starting article. " * 300,
            "outgoing": ["Start", "Middle", "Dead_end"],
        },
        "Middle": {"title": "Middle", "text": "Middle article.", "outgoing": ["Goal"]},
        "Goal": {"title": "Goal", "text": "PRIVATE target content.", "outgoing": ["Start"]},
        "Dead_end": {"title": "Dead end", "text": "PRIVATE dead-end content.", "outgoing": []},
    }
    path = tmp_path / "articles.json"
    path.write_text(json.dumps(articles), encoding="utf-8")
    environment = WikiGameEnv(path, max_clicks=2)
    yield environment
    environment.close()


def test_reset(env):
    """Reset the current article, route, click count, and accumulated reward."""
    env.reset(options={"start": "Start", "target": "Goal"})
    env.step(1)
    observation, info = env.reset(options={"start": "Start", "target": "Goal"})

    assert env.observation_space.contains(observation)
    assert observation["remaining_clicks"] == 2
    assert info["path"] == ["Start"]
    assert info["clicks"] == 0
    assert info["episode_return"] == 0
    assert not info["success"]


def test_get_llm_observation(env):
    """Expose full current text and available links without unvisited text."""
    env.reset(options={"start": "Start", "target": "Goal"})
    observation = env.get_llm_observation()

    assert observation["current_title"] == "Start"
    assert observation["target_title"] == "Goal"
    assert observation["text"] == "Starting article. " * 300
    assert observation["history"] == ["Start"]
    assert observation["actions"] == [
        {"id": 0, "title": "Start"},
        {"id": 1, "title": "Middle"},
        {"id": 2, "title": "Dead end"},
    ]
    assert "PRIVATE" not in json.dumps(observation)


def test_action_masks(env):
    """Update valid local action IDs when the current article changes."""
    env.reset(options={"start": "Start", "target": "Goal"})
    assert env.action_masks().tolist() == [True, True, True]

    env.step(1)
    assert env.action_masks().tolist() == [True, False, False]


def test_step_reaches_target(env):
    """Move through valid links and succeed on the last allowed click."""
    env.reset(options={"start": "Start", "target": "Goal"})
    observation, reward, terminated, truncated, info = env.step(1)

    assert env.observation_space.contains(observation)
    assert info["path"] == ["Start", "Middle"]
    assert reward == pytest.approx(-0.01)
    assert not terminated and not truncated

    observation, reward, terminated, truncated, info = env.step(0)
    assert env.observation_space.contains(observation)
    assert info["path"] == ["Start", "Middle", "Goal"]
    assert info["success"]
    assert info["clicks"] == 2
    assert info["episode_return"] == pytest.approx(0.98)
    assert reward == pytest.approx(0.99)
    assert terminated and not truncated
    assert not env.action_masks().any()


@pytest.mark.parametrize("actions, reason", [
    ([0, 0], "click_budget"),
    ([2], "dead_end"),
    ([1, 1], "invalid_action"),
])
def test_episode_failures(env, actions, reason):
    """End unsuccessful episodes with the failure penalty and correct reason."""
    env.reset(options={"start": "Start", "target": "Goal"})
    for action in actions:
        observation, reward, terminated, truncated, info = env.step(action)

    assert env.observation_space.contains(observation)
    assert terminated and not truncated
    assert not info["success"]
    assert info["reason"] == reason
    assert reward == pytest.approx(-1.01)


def test_step_requires_reset(env):
    """Require reset before the first step and after an episode finishes."""
    with pytest.raises(gym.error.ResetNeeded):
        env.step(0)

    env.reset(options={"start": "Start", "target": "Goal"})
    env.step(1)
    env.step(0)
    with pytest.raises(gym.error.ResetNeeded):
        env.step(0)
