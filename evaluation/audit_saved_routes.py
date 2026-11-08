import argparse
from collections import Counter
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _analyze_episode(episode, articles, budget):
    start, target = episode["start"], episode["target"]
    path, clicks = episode["path"], episode["clicks"]
    label = f"{start} -> {target}"
    if (not isinstance(start, str) or not isinstance(target, str)
            or start not in articles or target not in articles or start == target):
        raise ValueError(f"{label}: unknown or identical start and target.")
    if not articles[start]["outgoing"]:
        raise ValueError(f"{label}: starting article has no outgoing links.")
    if (not isinstance(path, list) or not path or path[0] != start
            or not all(isinstance(node, str) and node in articles for node in path)):
        raise ValueError(f"{label}: invalid path.")
    if type(clicks) is not int or not 1 <= clicks <= budget or clicks != len(path) - 1:
        raise ValueError(f"{label}: clicks/path length/budget disagree.")
    calls = episode.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != clicks:
        raise ValueError(f"{label}: provide one tool call per recorded move.")
    if target in path[:-1]:
        raise ValueError(f"{label}: recorded moves after reaching the target.")

    stack, visited = [start], {start}
    available, missed, revisits, self_links, backtracks = [], [], [], [], []
    tool_counts = Counter()
    for index, call in enumerate(calls):
        current, recorded = path[index:index + 2]
        outgoing = articles[current]["outgoing"]
        step_label = f"{label}, step {index + 1}"
        if (not isinstance(call, dict) or set(call) != {"name", "arguments"}
                or not isinstance(call["arguments"], dict)):
            raise ValueError(f"{step_label}: invalid tool-call format.")
        name, arguments = call["name"], call["arguments"]
        selected_id = None
        if name == "follow_link":
            selected_id = arguments.get("link_id")
            if (set(arguments) != {"link_id"} or type(selected_id) is not int
                    or not 0 <= selected_id < len(outgoing)):
                raise ValueError(f"{step_label}: invalid link_id.")
            chosen = outgoing[selected_id]
            stack.append(chosen)
        elif name == "go_back":
            if arguments:
                raise ValueError(f"{step_label}: go_back takes no arguments.")
            if len(stack) < 2:
                raise ValueError(f"{step_label}: Back is unavailable at the start.")
            stack.pop()
            chosen = stack[-1]
        else:
            raise ValueError(f"{step_label}: unknown tool {name!r}.")
        if chosen != recorded:
            raise ValueError(f"{step_label}: tool destination disagrees with the recorded path.")
        tool_counts[name] += 1
        if target in outgoing:
            event = {
                "step": index + 1, "remaining_clicks_before_move": budget - index,
                "current_id": current, "current_title": articles[current]["title"],
                "target_id": target, "target_title": articles[target]["title"],
                "expected_action_id": outgoing.index(target), "selected_action_id": selected_id,
                "selected_tool": name, "selected_article_id": chosen,
                "selected_title": articles[chosen]["title"], "link_count": len(outgoing),
                "selected_target": chosen == target,
            }
            available.append(event)
            if chosen != target:
                missed.append(event)
        if chosen in visited:
            revisits.append(index + 1)
        if name == "follow_link" and chosen == current:
            self_links.append(index + 1)
        if index >= 1 and chosen == path[index - 1] and chosen != current:
            backtracks.append(index + 1)
        visited.add(chosen)

    if episode.get("navigation_stack") != stack:
        raise ValueError(f"{label}: final navigation stack disagrees with tool replay.")
    success = path[-1] == target
    if episode.get("success") is not success:
        raise ValueError(f"{label}: success flag disagrees with the final article.")
    if not success and clicks != budget:
        raise ValueError(f"{label}: episode stopped before success or budget exhaustion.")
    reason = "success" if success else "click_budget"
    if episode.get("reason") != reason:
        raise ValueError(f"{label}: termination reason disagrees with the route.")
    if episode.get("terminated") is not True or episode.get("truncated") is not False:
        raise ValueError(f"{label}: termination flags disagree with the finite-horizon task.")
    expected_return = 1 + 0.1 * (budget - clicks) / budget if success else 0.0
    recorded_return = episode.get("episode_return")
    if (type(recorded_return) not in (int, float) or not math.isfinite(recorded_return)
            or not math.isclose(recorded_return, expected_return, abs_tol=1e-8, rel_tol=0)):
        raise ValueError(f"{label}: reward disagrees with the configured reward formula.")

    distance = episode.get("shortest_distance")
    if type(distance) is not int or distance < 1 or (success and distance > clicks):
        raise ValueError(f"{label}: invalid shortest_distance.")
    efficiency = episode.get("path_efficiency")
    if success:
        if (type(efficiency) not in (int, float) or not math.isfinite(efficiency)
                or not math.isclose(efficiency, distance / clicks, abs_tol=1e-8, rel_tol=0)):
            raise ValueError(f"{label}: path efficiency disagrees with the route.")
    elif efficiency is not None:
        raise ValueError(f"{label}: a failed route must have null path efficiency.")

    return {
        "start": start, "start_title": articles[start]["title"],
        "target": target, "target_title": articles[target]["title"],
        "success": success, "clicks": clicks, "episode_return": recorded_return,
        "reason": reason, "path": path, "navigation_stack": stack,
        "follow_link_calls": tool_counts["follow_link"], "go_back_calls": tool_counts["go_back"],
        "direct_target_opportunities": len(available),
        "missed_direct_target_count": len(missed),
        "unique_articles_with_missed_target": len({e["current_id"] for e in missed}),
        "missed_direct_target_steps": missed,
        "target_clicked_steps": [e["step"] for e in available if e["selected_target"]],
        "revisit_count": len(revisits), "revisit_steps": revisits,
        "self_link_count": len(self_links), "self_link_steps": self_links,
        "immediate_backtrack_count": len(backtracks), "immediate_backtrack_steps": backtracks,
        "most_visited": [
            {"article_id": node, "title": articles[node]["title"], "visits": count}
            for node, count in Counter(path).most_common(4)
        ],
    }


def analyze_routes(report, articles):
    """Replay current-format tool traces; fail rather than accept inconsistent logs."""
    if not isinstance(report, dict) or not isinstance(report.get("settings", {}), dict):
        raise ValueError("Results and settings must be JSON objects.")
    episodes = report.get("episodes")
    settings = report.get("settings", {})
    budget = settings.get("max_clicks")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Results need a nonempty episodes list.")
    if type(budget) is not int or budget < 1:
        raise ValueError("Results settings must contain a positive max_clicks.")
    if not isinstance(articles, dict) or not articles:
        raise ValueError("Article graph must be a nonempty dictionary.")
    results = [_analyze_episode(episode, articles, budget) for episode in episodes]
    failures = [episode for episode in results if not episode["success"]]
    opportunities = sum(e["direct_target_opportunities"] for e in results)
    clicked = sum(len(e["target_clicked_steps"]) for e in results)
    summary = {
        "episodes": len(results), "successes": len(results) - len(failures), "failures": len(failures),
        "total_clicks": sum(e["clicks"] for e in results),
        "follow_link_calls": sum(e["follow_link_calls"] for e in results),
        "go_back_calls": sum(e["go_back_calls"] for e in results),
        "episodes_with_back": sum(e["go_back_calls"] > 0 for e in results),
        "all_recorded_transitions_and_returns_valid": True,
        "direct_target_opportunities": opportunities,
        "target_clicked_count": clicked,
        "missed_direct_target_count": sum(e["missed_direct_target_count"] for e in results),
        "direct_target_selection_rate_percent": 100 * clicked / opportunities if opportunities else None,
        "episodes_with_missed_direct_targets": sum(bool(e["missed_direct_target_count"]) for e in results),
        "failed_episodes_with_missed_direct_targets": sum(bool(e["missed_direct_target_count"]) for e in failures),
        "failed_episodes_without_direct_target_opportunity": sum(e["direct_target_opportunities"] == 0 for e in failures),
        "episodes_with_revisits": sum(bool(e["revisit_count"]) for e in results),
        "failed_episodes_with_revisits": sum(bool(e["revisit_count"]) for e in failures),
        "total_revisits": sum(e["revisit_count"] for e in results),
        "self_link_moves": sum(e["self_link_count"] for e in results),
        "immediate_backtracks": sum(e["immediate_backtrack_count"] for e in results),
        "termination_reasons": dict(Counter(e["reason"] for e in results)),
    }
    return {
        "summary": summary, "settings": settings, "episodes": results,
        "interpretation": {
            "missed_direct_target": "Target was a visible link, but follow_link chose another ID or go_back was selected. This does not distinguish copying errors from navigation choices.",
            "revisit": "A move to any previously visited article, including Back and self-links. Revisits can be useful.",
            "self_link": "A follow_link call whose destination is the current article; Back across a self-link is counted separately as go_back.",
            "immediate_backtrack": "A distinct-node A -> B -> A pattern, regardless of which tools were used. This differs from the count of go_back calls.",
            "overlap": "Missed-target, revisit, self-link, and backtrack counts overlap; they are not disjoint failure causes.",
            "opportunities": "Repeated visits with the target available are repeated opportunities, not independent samples.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Saved evaluation JSON.")
    parser.add_argument("--articles", type=Path, default=ROOT / "data/processed/articles.json")
    parser.add_argument("--output", type=Path, required=True, help="Destination audit JSON.")
    args = parser.parse_args()
    if args.output.resolve() in {args.results.resolve(), args.articles.resolve()}:
        parser.error("Output must not overwrite either input file.")
    try:
        report = json.loads(args.results.read_text(encoding="utf-8"))
        articles = json.loads(args.articles.read_text(encoding="utf-8"))
        audit = analyze_routes(report, articles)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        parser.exit(1, f"Route audit failed: {error}\n")
    audit["inputs"] = {"results": str(args.results), "articles": str(args.articles)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit["summary"], indent=2))
    for episode in audit["episodes"]:
        for event in episode["missed_direct_target_steps"]:
            selected = "go_back" if event["selected_tool"] == "go_back" else f"follow_link({event['selected_action_id']})"
            print(f"MISSED TARGET | {episode['start_title']} -> {episode['target_title']} | "
                  f"step {event['step']} at {event['current_title']}: "
                  f"expected {event['expected_action_id']}: {event['target_title']}, "
                  f"selected {selected} -> {event['selected_title']}")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
