import json

import gymnasium as gym
import pytest

from environment.environment import WikiGameEnv


@pytest.fixture
def env(tmp_path, request):
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
    environment = WikiGameEnv(path, max_clicks=getattr(request, "param", 2))
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
    assert info["navigation_stack"] == ["Start"]
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
    assert observation["navigation_stack"] == ["Start"]
    assert not observation["back_available"]
    assert observation["back_destination"] is None
    assert observation["back_action_id"] is None
    assert observation["actions"] == [
        {"id": 0, "title": "Start"},
        {"id": 1, "title": "Middle"},
        {"id": 2, "title": "Dead end"},
    ]
    assert "PRIVATE" not in json.dumps(observation)


def test_action_masks(env):
    """Update valid local action IDs when the current article changes."""
    env.reset(options={"start": "Start", "target": "Goal"})
    assert env.action_masks().tolist() == [True, True, True, False]

    env.step(1)
    assert env.action_masks().tolist() == [True, False, False, True]


def test_step_reaches_target(env):
    """Move through valid links and succeed on the last allowed click."""
    env.reset(options={"start": "Start", "target": "Goal"})
    observation, reward, terminated, truncated, info = env.step(1)

    assert env.observation_space.contains(observation)
    assert info["path"] == ["Start", "Middle"]
    assert reward == 0.0
    assert info["episode_return"] == 0.0
    assert not terminated and not truncated

    observation, reward, terminated, truncated, info = env.step(0)
    assert env.observation_space.contains(observation)
    assert info["path"] == ["Start", "Middle", "Goal"]
    assert info["success"]
    assert info["clicks"] == 2
    assert info["episode_return"] == pytest.approx(1.0)
    assert reward == pytest.approx(1.0)
    assert terminated and not truncated
    assert not env.action_masks().any()
    assert not env.get_llm_observation()["back_available"]


@pytest.mark.parametrize("actions, expected_reward", [
    ([1], 1.05),
    ([0, 1], 1.0),
])
def test_success_efficiency_bonus(env, actions, expected_reward):
    """Give shorter successful routes a larger reward within the same budget."""
    env.reset(options={"start": "Start", "target": "Middle"})
    for index, action in enumerate(actions):
        observation, reward, terminated, truncated, info = env.step(action)
        if index < len(actions) - 1:
            assert reward == 0.0
            assert not terminated and not truncated

    assert env.observation_space.contains(observation)
    assert info["success"]
    assert info["reason"] == "success"
    assert info["clicks"] == len(actions)
    assert reward == pytest.approx(expected_reward)
    assert info["episode_return"] == pytest.approx(expected_reward)
    assert terminated and not truncated


@pytest.mark.parametrize("actions, reason", [
    ([0, 0], "click_budget"),
    ([2, 3], "click_budget"),
    ([3], "invalid_action"),
    ([1, 1], "invalid_action"),
])
def test_episode_failures(env, actions, reason):
    """Give zero reward throughout failed episodes and report the correct reason."""
    env.reset(options={"start": "Start", "target": "Goal"})
    for action in actions:
        observation, reward, terminated, truncated, info = env.step(action)
        assert reward == 0.0

    assert env.observation_space.contains(observation)
    assert terminated and not truncated
    assert not info["success"]
    assert info["reason"] == reason
    assert info["episode_return"] == 0.0


def test_step_requires_reset(env):
    """Require reset before the first step and after an episode finishes."""
    with pytest.raises(gym.error.ResetNeeded):
        env.step(0)

    env.reset(options={"start": "Start", "target": "Goal"})
    env.step(1)
    env.step(0)
    with pytest.raises(gym.error.ResetNeeded):
        env.step(0)


def test_back_escapes_a_dead_end(env):
    env.reset(options={"start": "Start", "target": "Goal"})
    _, reward, terminated, truncated, _ = env.step(2)

    assert reward == 0 and not terminated and not truncated
    assert env.action_masks().tolist() == [False, False, False, True]
    observation = env.get_llm_observation()
    assert observation["actions"] == []
    assert observation["back_available"]
    assert observation["back_destination"] == "Start"
    assert observation["back_action_id"] == env.back_action

    observation, reward, terminated, truncated, info = env.step(env.back_action)
    assert env.observation_space.contains(observation)
    assert terminated and not truncated and reward == 0
    assert info["reason"] == "click_budget"
    assert info["path"] == ["Start", "Dead_end", "Start"]
    assert info["navigation_stack"] == ["Start"]
    assert info["clicks"] == 2


@pytest.mark.parametrize("env", [8], indirect=True)
def test_branch_after_back_keeps_complete_route(env):
    env.reset(options={"start": "Start", "target": "Goal"})
    actions = [1, env.back_action, 2, env.back_action, 1, 0]
    for action in actions:
        observation, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(observation)

    assert info["path"] == ["Start", "Middle", "Start", "Dead_end", "Start", "Middle", "Goal"]
    assert info["navigation_stack"] == ["Start", "Middle", "Goal"]
    assert info["clicks"] == len(actions)
    assert terminated and not truncated and info["success"]
    assert reward == pytest.approx(1.025)
    visible = env.get_llm_observation()
    assert len(visible["history"]) == 7  # Full route, beyond the five-entry numeric history.
    assert visible["navigation_stack"] == ["Start", "Middle", "Goal"]
    assert visible["actions"] == [] and not visible["back_available"]


@pytest.mark.parametrize("env", [8], indirect=True)
def test_repeated_back_tracks_stack_entries_including_self_links(env):
    env.reset(options={"start": "Start", "target": "Goal"})
    for action in [0, 0, 1]:
        env.step(action)
    assert env.navigation_stack == ["Start", "Start", "Start", "Middle"]
    for expected_stack in [["Start", "Start", "Start"], ["Start", "Start"], ["Start"]]:
        observation, reward, terminated, truncated, info = env.step(env.back_action)
        assert env.observation_space.contains(observation)
        assert info["navigation_stack"] == expected_stack
        assert reward == 0 and not terminated and not truncated
    assert info["clicks"] == 6
    assert len(info["path"]) == 7
    assert not env.action_masks()[env.back_action]


@pytest.mark.parametrize("env", [4], indirect=True)
def test_back_and_reset_do_not_mutate_previous_observations(env):
    env.reset(options={"start": "Start", "target": "Goal"})
    env.step(1)
    previous_observation, previous_info = env.get_llm_observation(), env.get_info()
    env.step(env.back_action)
    env.reset(options={"start": "Middle", "target": "Goal"})

    assert previous_info["path"] == ["Start", "Middle"]
    assert previous_info["navigation_stack"] == ["Start", "Middle"]
    assert previous_observation["history"] == ["Start", "Middle"]
    assert previous_observation["back_destination"] == "Start"
    assert env.navigation_stack == env.history == ["Middle"]
    assert env.clicks == 0


def test_out_of_range_action_does_not_change_state(env):
    env.reset(options={"start": "Start", "target": "Goal"})
    before = env.get_info()
    with pytest.raises(ValueError, match="outside the action space"):
        env.step(env.action_space.n)
    assert env.get_info() == before
