import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

from environment.environment import WikiGameEnv
from environment.episode_runner import EpisodeRunner
from evaluation.audit_saved_routes import analyze_routes
from policy.llm_policy import WikiGamePolicy


def evaluate(runner, tasks):
    """Run greedy episodes and return per-game results and summary metrics.

    Args:
        runner: An EpisodeRunner with an existing environment and policy.
        tasks: A non-empty list of dictionaries containing start, target,
            and shortest_distance.

    Returns:
        A dictionary containing summary metrics and episode outcomes. Success
        rate is a percentage. Path efficiency is shortest_distance / clicks,
        averaged over successful episodes; 1 means an optimal route. Successful
        path length and path efficiency are None when no games succeed.
        Full observations and article text are not saved.
    """
    if not tasks:
        raise ValueError("Provide at least one task to evaluate.")
    for task in tasks:
        distance = task.get("shortest_distance")
        if type(distance) is not int or distance < 1:
            raise ValueError("Each task needs a positive integer shortest_distance.")

    results = []
    success_count = 0
    with tqdm(tasks, desc="Evaluating tasks", unit="task", dynamic_ncols=True) as progress:
        for task in progress:
            episode = runner.run_episode(task["start"], task["target"], deterministic=True)
            result = {
                "start": episode["start"],
                "target": episode["target"],
                "path": episode["path"],
                "tool_calls": [step["tool_call"] for step in episode["steps"]],
                "navigation_stack": episode["navigation_stack"],
                "success": episode["success"],
                "clicks": episode["clicks"],
                "episode_return": episode["episode_return"],
                "reason": episode["reason"],
                "terminated": episode["terminated"],
                "truncated": episode["truncated"],
                "shortest_distance": task["shortest_distance"],
                "path_efficiency": None,
            }
            if result["success"]:
                if result["clicks"] < task["shortest_distance"]:
                    raise ValueError("A successful route is shorter than shortest_distance.")
                result["path_efficiency"] = task["shortest_distance"] / result["clicks"]
                success_count += 1
            results.append(result)
            progress.set_postfix(success_rate=f"{100 * success_count / len(results):.1f}%",
                                 refresh=False)

    successes = [result for result in results if result["success"]]
    average_clicks = None
    path_efficiency = None
    if successes:
        average_clicks = sum(result["clicks"] for result in successes) / len(successes)
        path_efficiency = sum(result["path_efficiency"] for result in successes) / len(successes)

    success_rate = 100 * len(successes) / len(results)
    summary = {
        "num_episodes": len(results),
        "successes": len(successes),
        "success_rate_percent": success_rate,
        "average_return": sum(result["episode_return"] for result in results) / len(results),
        "successful_path_length": average_clicks,
        "path_efficiency": path_efficiency,
    }
    report = {
        "summary": summary, "episodes": results,
        "settings": {"max_clicks": runner.env.max_clicks},
    }
    report["route_audit"] = analyze_routes(report, runner.env.articles)["summary"]
    return report


def main():
    """Evaluate validation tasks and save results to evaluation/results/baseline.json."""
    parser = argparse.ArgumentParser(description="Evaluate the Qwen Wiki Game baseline.")
    parser.add_argument("--tasks", type=Path, required=True,
                        help="Validation JSON tasks with start, target, and shortest_distance.")
    parser.add_argument("--limit", type=int,
                        help="Evaluate up to this many randomly selected tasks, using seed 42.")
    parser.add_argument("--max-clicks", type=int, default=20,
                        help="Click budget per game (default: 20).")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the evaluation results JSON.",
    )
    args = parser.parse_args()
    if args.max_clicks < 1 or (args.limit is not None and args.limit < 1):
        parser.error("The click budget and task limit must be positive.")

    with args.tasks.open(encoding="utf-8") as file:
        tasks = json.load(file)
    if not isinstance(tasks, list) or not tasks:
        parser.error("The tasks file must contain a non-empty JSON list.")
    if args.limit is not None:
        tasks = random.Random(42).sample(tasks, min(args.limit, len(tasks)))

    env = WikiGameEnv(max_clicks=args.max_clicks)
    try:
        for task in tasks:
            env.validate_pair(task["start"], task["target"])

        policy = WikiGamePolicy(max_input_tokens=4096)
        runner = EpisodeRunner(env, policy)
        report = evaluate(runner, tasks)
        report["settings"].update({
            "model": policy.model.name_or_path,
            "tasks_file": str(args.tasks),
            "deterministic": True,
            "sample_seed": 42 if args.limit is not None else None,
            "max_input_tokens": policy.max_input_tokens,
            "action_format": "qwen_tool_call",
            "enable_thinking": False,
            "system_prompt": policy.system_prompt,
        })

        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
        print(json.dumps(report["summary"], indent=2))
        print(f"Results saved to {output_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
