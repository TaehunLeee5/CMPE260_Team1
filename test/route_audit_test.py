from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from evaluation.audit_saved_routes import analyze_routes


def follow(link_id):
    return {"name": "follow_link", "arguments": {"link_id": link_id}}


def back():
    return {"name": "go_back", "arguments": {}}


def make_report(path, calls, stack, budget=6):
    clicks = len(calls)
    success = path[-1] == "T"
    return {
        "settings": {"max_clicks": budget},
        "episodes": [{
            "start": "A", "target": "T", "path": path,
            "tool_calls": calls, "navigation_stack": stack,
            "clicks": clicks, "success": success,
            "episode_return": 1 + 0.1 * (budget - clicks) / budget if success else 0.0,
            "reason": "success" if success else "click_budget",
            "terminated": True, "truncated": False,
            "shortest_distance": 1, "path_efficiency": 1 / clicks if success else None,
        }],
    }


@pytest.fixture
def articles():
    return {
        "A": {"title": "Article A", "text": "A", "outgoing": ["A", "B", "C", "T"]},
        "B": {"title": "Article B", "text": "B", "outgoing": ["A", "T"]},
        "C": {"title": "Article C", "text": "C", "outgoing": []},
        "T": {"title": "Target", "text": "T", "outgoing": ["A"]},
    }


@pytest.fixture
def report():
    return make_report(
        ["A", "C", "A", "B", "T"],
        [follow(2), back(), follow(1), follow(1)], ["A", "B", "T"],
    )


@pytest.mark.parametrize("budget", [4, 6])
def test_back_without_reverse_link_is_valid(report, articles, budget):
    report["settings"]["max_clicks"] = budget
    report["episodes"][0]["episode_return"] = 1 + 0.1 * (budget - 4) / budget
    original = deepcopy(report)
    audit = analyze_routes(report, articles)
    summary = audit["summary"]
    assert summary["all_recorded_transitions_and_returns_valid"]
    assert summary["total_clicks"] == 4
    assert summary["follow_link_calls"] == 3
    assert summary["go_back_calls"] == summary["episodes_with_back"] == 1
    assert summary["direct_target_opportunities"] == 3
    assert summary["missed_direct_target_count"] == 2
    assert summary["target_clicked_count"] == 1
    assert summary["direct_target_selection_rate_percent"] == pytest.approx(100 / 3)
    assert summary["total_revisits"] == summary["immediate_backtracks"] == 1
    assert audit["episodes"][0]["navigation_stack"] == ["A", "B", "T"]
    assert report == original
    assert json.loads(json.dumps(audit)) == audit


def test_back_can_miss_a_visible_target(articles):
    report = make_report(["A", "B", "A", "T"], [follow(1), back(), follow(3)], ["A", "T"])
    audit = analyze_routes(report, articles)
    event = audit["episodes"][0]["missed_direct_target_steps"][1]
    assert event["current_id"] == "B"
    assert event["selected_tool"] == "go_back"
    assert event["selected_action_id"] is None
    assert event["selected_article_id"] == "A"
    assert audit["summary"]["missed_direct_target_count"] == 2


@pytest.mark.parametrize("use_back", [False, True])
def test_tool_trace_distinguishes_identical_routes(articles, use_back):
    report = make_report(
        ["A", "B", "A", "B", "T"],
        [follow(1), follow(0), back() if use_back else follow(1), follow(1)],
        ["A", "B", "T"] if use_back else ["A", "B", "A", "B", "T"],
    )
    summary = analyze_routes(report, articles)["summary"]
    assert summary["go_back_calls"] == int(use_back)
    assert summary["immediate_backtracks"] == 2


def test_back_across_self_link_is_not_a_second_self_link(articles):
    report = make_report(["A", "A", "A", "T"], [follow(0), back(), follow(3)], ["A", "T"])
    summary = analyze_routes(report, articles)["summary"]
    assert summary["self_link_moves"] == summary["go_back_calls"] == 1
    assert summary["total_revisits"] == 2
    assert summary["immediate_backtracks"] == 0


def test_repeated_back_replays_duplicate_stack_entries(articles):
    report = make_report(
        ["A", "B", "A", "B", "A", "T"],
        [follow(1), follow(0), back(), back(), follow(3)], ["A", "T"],
    )
    assert analyze_routes(report, articles)["summary"]["go_back_calls"] == 2


def test_summary_aggregates_success_and_budget_failure(report, articles):
    failure = make_report(["A"] * 7, [follow(0)] * 6, ["A"] * 7)["episodes"][0]
    report["episodes"].append(failure)
    summary = analyze_routes(report, articles)["summary"]
    assert summary["episodes"] == 2
    assert summary["successes"] == summary["failures"] == 1
    assert summary["total_clicks"] == 10
    assert summary["follow_link_calls"] == 9
    assert summary["go_back_calls"] == 1
    assert summary["missed_direct_target_count"] == 8
    assert summary["termination_reasons"] == {"success": 1, "click_budget": 1}


@pytest.mark.parametrize("field,value", [
    ("clicks", 3), ("success", False), ("episode_return", 0),
    ("episode_return", float("nan")), ("reason", "dead_end"),
    ("terminated", False), ("truncated", True), ("navigation_stack", ["A", "T"]),
    ("shortest_distance", 0), ("path_efficiency", 0.5), ("tool_calls", []),
    ("path", ["A", "C", "T", "B", "T"]),
])
def test_corrupted_episode_fields_rejected(report, articles, field, value):
    report["episodes"][0][field] = value
    with pytest.raises(ValueError):
        analyze_routes(report, articles)


@pytest.mark.parametrize("link_id", [-1, True, 2.0, 999])
def test_invalid_link_id_rejected(report, articles, link_id):
    report["episodes"][0]["tool_calls"][0] = follow(link_id)
    with pytest.raises(ValueError, match="invalid link_id"):
        analyze_routes(report, articles)


@pytest.mark.parametrize("call", [
    {"name": "search", "arguments": {}},
    {"name": "follow_link", "arguments": {}},
    {"name": "follow_link", "arguments": {"link_id": 2, "extra": 1}},
    {"name": "follow_link", "arguments": [2]},
    {"arguments": {"link_id": 2}},
])
def test_malformed_tool_call_rejected(report, articles, call):
    report["episodes"][0]["tool_calls"][0] = call
    with pytest.raises(ValueError):
        analyze_routes(report, articles)


def test_back_at_start_rejected(report, articles):
    report["episodes"][0]["tool_calls"][0] = back()
    with pytest.raises(ValueError, match="Back is unavailable"):
        analyze_routes(report, articles)


def test_back_arguments_rejected(report, articles):
    report["episodes"][0]["tool_calls"][1]["arguments"] = {"link_id": 0}
    with pytest.raises(ValueError, match="takes no arguments"):
        analyze_routes(report, articles)


@pytest.mark.parametrize("step", [0, 2])
def test_wrong_forward_destination_rejected(report, articles, step):
    report["episodes"][0]["tool_calls"][step] = follow(3)
    with pytest.raises(ValueError, match="destination disagrees"):
        analyze_routes(report, articles)


def test_wrong_back_destination_rejected(report, articles):
    report["episodes"][0]["path"][2] = "B"
    with pytest.raises(ValueError, match="destination disagrees"):
        analyze_routes(report, articles)


def test_missing_tool_trace_is_not_inferred_from_path(report, articles):
    del report["episodes"][0]["tool_calls"]
    with pytest.raises(ValueError, match="one tool call per recorded move"):
        analyze_routes(report, articles)


def test_premature_failure_is_rejected(articles):
    report = make_report(["A", "B"], [follow(1)], ["A", "B"])
    with pytest.raises(ValueError, match="stopped before"):
        analyze_routes(report, articles)


def test_no_direct_target_opportunities(articles):
    report = {
        "settings": {"max_clicks": 2},
        "episodes": [{
            "start": "B", "target": "C", "path": ["B", "A", "B"],
            "tool_calls": [follow(0), back()], "navigation_stack": ["B"],
            "clicks": 2, "success": False, "episode_return": 0,
            "reason": "click_budget", "terminated": True, "truncated": False,
            "shortest_distance": 2, "path_efficiency": None,
        }],
    }
    # A normally links to C; use a graph with the goal reachable via T instead.
    articles["A"]["outgoing"] = ["A", "B", "T"]
    articles["T"]["outgoing"] = ["C"]
    report["episodes"][0]["shortest_distance"] = 3
    summary = analyze_routes(report, articles)["summary"]
    assert summary["direct_target_opportunities"] == 0
    assert summary["direct_target_selection_rate_percent"] is None
    assert summary["failed_episodes_without_direct_target_opportunity"] == 1


def test_evaluation_and_standalone_audit_agree(tmp_path, articles, report):
    from environment.environment import WikiGameEnv
    from environment.episode_runner import EpisodeRunner
    from evaluation.evaluate import evaluate

    path = tmp_path / "articles.json"
    path.write_text(json.dumps(articles), encoding="utf-8")
    calls = iter(report["episodes"][0]["tool_calls"])

    def choose(observation, deterministic=False):
        call = next(calls)
        action = observation["back_action_id"] if call["name"] == "go_back" else call["arguments"]["link_id"]
        return {"action": action, "tool_call": call, "log_prob": 0.0}

    env = WikiGameEnv(path, max_clicks=6)
    result = evaluate(EpisodeRunner(env, SimpleNamespace(choose_action=choose)), [
        {"start": "A", "target": "T", "shortest_distance": 1},
    ])
    assert result["route_audit"] == analyze_routes(result, articles)["summary"]
    assert result["route_audit"]["go_back_calls"] == 1


def test_cli_and_input_overwrite_protection(tmp_path, report, articles):
    inputs = tmp_path / "results.json"
    graph = tmp_path / "articles.json"
    output = tmp_path / "audit.json"
    inputs.write_text(json.dumps(report), encoding="utf-8")
    graph.write_text(json.dumps(articles), encoding="utf-8")
    args = [sys.executable, "-m", "evaluation.audit_saved_routes",
            "--results", str(inputs), "--articles", str(graph), "--output"]
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run([*args, str(output)], cwd=root, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["summary"] == analyze_routes(report, articles)["summary"]
    assert "follow_link(2)" in run.stdout
    for destination in [inputs, graph]:
        before = destination.read_bytes()
        run = subprocess.run([*args, str(destination)], cwd=root, capture_output=True, text=True)
        assert run.returncode != 0
        assert "must not overwrite" in run.stderr
        assert destination.read_bytes() == before
