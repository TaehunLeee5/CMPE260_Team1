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



@pytest.fixture
def progress_files(tmp_path):
    """The fixture graph plus its shortest-path matrix, in articles.json order."""
    articles = {
        "Start": {"title": "Start", "text": "Start.", "outgoing": ["Start", "Middle", "Dead_end"]},
        "Middle": {"title": "Middle", "text": "Middle.", "outgoing": ["Goal"]},
        "Goal": {"title": "Goal", "text": "Goal.", "outgoing": ["Start"]},
        "Dead_end": {"title": "Dead end", "text": "Dead end.", "outgoing": []},
    }
    paths = {name: tmp_path / name for name in ["articles.json", "matrix.txt", "order.json"]}
    paths["articles.json"].write_text(json.dumps(articles), encoding="utf-8")
    # Rows: from Start, Middle, Goal, Dead_end; columns in the same order.
    paths["matrix.txt"].write_text("# comment\n0121\n2013\n1202\n___0\n", encoding="utf-8")
    paths["order.json"].write_text(json.dumps(list(articles)), encoding="utf-8")
    return paths


def progress_env(paths, max_clicks):
    return WikiGameEnv(paths["articles.json"], max_clicks=max_clicks, reward="progress",
                       distance_matrix_path=paths["matrix.txt"], article_order_path=paths["order.json"])


@pytest.mark.parametrize("action, expected", [
    (1, 0.25 * (2 - 1) / 2),  # Middle is one click from Goal: half of the possible progress.
    (0, 0.0),                 # Self-link: still two clicks away.
    (2, 0.0),                 # Dead end: Goal is unreachable from there.
])
def test_progress_credit_for_failed_routes(progress_files, action, expected):
    env = progress_env(progress_files, max_clicks=1)
    env.reset(options={"start": "Start", "target": "Goal"})
    _, reward, terminated, _, info = env.step(action)
    assert terminated and info["reason"] == "click_budget"
    assert reward == pytest.approx(expected) and info["episode_return"] == pytest.approx(expected)
    assert env.route_return(info["path"], "Goal", 1, False) == pytest.approx(expected)


def test_progress_is_paid_when_reached_and_success_gets_full_credit(progress_files):
    env = progress_env(progress_files, max_clicks=2)
    env.reset(options={"start": "Start", "target": "Goal"})
    assert env.step(1)[1] == pytest.approx(0.125)  # New closest distance: paid immediately.
    _, reward, terminated, _, info = env.step(0)    # Middle -> Goal.
    assert terminated and info["success"]
    assert reward == pytest.approx(0.125 + 1.0)    # Last bit of progress plus the success reward.
    assert info["episode_return"] == pytest.approx(1.25)
    assert env.route_return(info["path"], "Goal", 2, True) == pytest.approx(1.25)


def test_revisits_and_lingering_earn_nothing(progress_files):
    env = progress_env(progress_files, max_clicks=3)
    env.reset(options={"start": "Start", "target": "Goal"})
    rewards = [env.step(1)[1], env.step(env.back_action)[1]]
    env.reset(options={"start": "Start", "target": "Goal"})
    env.step(1)                       # Start -> Middle (+0.125)
    env.step(env.back_action)         # back to Start
    _, reward, terminated, _, info = env.step(1)  # Middle again: not a new closest distance.
    assert rewards == pytest.approx([0.125, 0.0]) and reward == 0.0
    assert terminated and info["episode_return"] == pytest.approx(0.125)


def test_every_success_outranks_every_failure(progress_files):
    env = progress_env(progress_files, max_clicks=20)
    slowest_success = env.route_return(["Start"] * 20 + ["Goal"], "Goal", 20, True)
    best_failure = env.route_return(["Start", "Middle"], "Goal", 20, False)
    assert slowest_success == pytest.approx(1.25)
    assert best_failure == pytest.approx(0.125)
    assert best_failure < env.PROGRESS_WEIGHT < slowest_success
    assert env.shortest_distance("Dead_end", "Goal") is None
    assert env.shortest_distance("Middle", "Dead_end") == 3


def test_distance_matrix_must_match_the_link_graph(progress_files):
    # Valid format, but Start -> Goal is really 2 clicks (via Middle), not 3.
    progress_files["matrix.txt"].write_text("0131\n2013\n1202\n___0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the link graph"):
        progress_env(progress_files, 1)


def test_success_reward_needs_no_matrix(env):
    """The default reward loads no matrix, and failures still earn exactly zero."""
    env.reset(options={"start": "Start", "target": "Goal"})
    env.step(1)  # Start -> Middle, one click from Goal: no credit under the default reward.
    _, reward, terminated, _, info = env.step(env.back_action)
    assert env.reward_mode == "success" and terminated and not info["success"] and reward == 0.0
    with pytest.raises(ValueError, match="only loaded"):
        env.shortest_distance("Start", "Goal")


def test_progress_reward_rejects_misaligned_inputs(progress_files):
    progress_files["order.json"].write_text(json.dumps(["Middle", "Start", "Goal", "Dead_end"]), encoding="utf-8")
    with pytest.raises(ValueError, match="article order"):
        progress_env(progress_files, 1)
    progress_files["order.json"].write_text(json.dumps(["Start", "Middle", "Goal", "Dead_end"]), encoding="utf-8")
    progress_files["matrix.txt"].write_text("0121\n2013\n1202\n", encoding="utf-8")
    with pytest.raises(ValueError, match="number of rows"):
        progress_env(progress_files, 1)
    progress_files["matrix.txt"].write_text("0121\n2013\n1202\n__0_\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid distance matrix row"):
        progress_env(progress_files, 1)
    with pytest.raises(ValueError, match="reward must be"):
        WikiGameEnv(progress_files["articles.json"], reward="dense")



def test_progress_episodes_pass_the_training_audit(progress_files):
    """Run real episodes through the runner and the training report path."""
    from environment.episode_runner import EpisodeRunner
    from evaluation.audit_saved_routes import analyze_routes
    from training.grpo import collect_groups, compact_report

    class TowardMiddle:
        def choose_action(self, observation, deterministic=False):
            action = next(a for a in observation["actions"] if a["title"] == "Middle")
            return {"action": action["id"], "log_prob": 0.0,
                    "tool_call": {"name": "follow_link", "arguments": {"title": "Middle"}}}

    env = progress_env(progress_files, max_clicks=1)
    runner = EpisodeRunner(env, TowardMiddle())
    groups = collect_groups(runner, [{"start": "Start", "target": "Goal", "shortest_distance": 2}],
                            rollouts=2, seed=None)
    for episode in groups[0]:
        assert not episode["success"] and episode["episode_return"] == pytest.approx(0.125)
    report = compact_report(groups, env.max_clicks, env.reward_mode)
    assert report["settings"]["reward"] == "progress"
    audit = analyze_routes(report, env.articles, env.route_return)["summary"]
    assert audit["all_recorded_transitions_and_returns_valid"] and audit["failures"] == 2



def test_link_context_fields_reach_the_observation(tmp_path):
    articles = {
        "Start": {"title": "Start", "text": "Full text.", "outgoing": ["Goal", "Other"],
                  "lead": "Start is an article.", "link_order": ["Other", "Goal"],
                  "link_context": {"Goal": "Start links to Goal here."}},
        "Goal": {"title": "Goal", "text": "", "outgoing": ["Start"]},
        "Other": {"title": "Other", "text": "", "outgoing": ["Start"]},
    }
    path = tmp_path / "articles.json"
    path.write_text(json.dumps(articles), encoding="utf-8")
    env = WikiGameEnv(path, max_clicks=3)
    env.reset(options={"start": "Start", "target": "Goal"})
    observation = env.get_llm_observation()
    assert observation["lead"] == "Start is an article."
    assert observation["actions"] == [
        {"id": 0, "title": "Goal", "position": 1, "context": "Start links to Goal here."},
        {"id": 1, "title": "Other", "position": 0, "context": None},
    ]
    env.step(0)  # Goal has no context fields: the observation has none either.
