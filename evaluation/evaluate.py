import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import mean, pstdev

from tqdm import tqdm
from transformers import set_seed

from environment.environment import WikiGameEnv
from environment.episode_runner import EpisodeRunner
from evaluation.audit_saved_routes import analyze_routes
from policy.llm_policy import PROMPT_STYLES, WikiGamePolicy


def evaluate(runner, tasks, *, deterministic=True, rollouts=1, seed=42):
    """Run greedy evaluation or grouped sampled rollouts without policy updates.

    Args:
        runner: An EpisodeRunner with an existing environment and policy.
        tasks: A non-empty list of dictionaries containing start, target,
            and shortest_distance.
        deterministic: Use greedy actions; False enables policy sampling.
        rollouts: Routes per task; values above one require sampling.
        seed: Seed once before sampled collection, never once per route.

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

    if type(rollouts) is not int or rollouts < 1:
        raise ValueError("rollouts must be a positive integer.")
    if deterministic and rollouts > 1:
        raise ValueError("Multiple rollouts require sampled mode.")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer between 0 and 2**32 - 1.")
    pairs = {(task["start"], task["target"]) for task in tasks}
    if len(pairs) != len(tasks):
        raise ValueError("Tasks must contain unique start-target pairs; use rollouts for repeats.")
    if not deterministic:
        set_seed(seed)

    results = []
    success_count = 0
    with tqdm(total=len(tasks) * rollouts, desc="Evaluating" if deterministic else "Sampling",
              unit="episode", dynamic_ncols=True) as progress:
        for task_index, task in enumerate(tasks, 1):
            for rollout_index in range(1, rollouts + 1):
                episode = runner.run_episode(task["start"], task["target"], deterministic=deterministic)
                result = {
                    "task_index": task_index,
                    "rollout_index": rollout_index,
                    "start": episode["start"],
                    "target": episode["target"],
                    "path": episode["path"],
                    "tool_calls": [step["tool_call"] for step in episode["steps"]],
                    "action_log_probs": [step["log_prob"] for step in episode["steps"]],
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
                progress.update(1)

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
        "settings": {
            "max_clicks": runner.env.max_clicks, "reward": runner.env.reward_mode,
            "deterministic": deterministic,
            "rollouts_per_task": rollouts, "rollout_seed": seed if not deterministic else None,
        },
    }
    report["route_audit"] = analyze_routes(report, runner.env.articles, runner.env.route_return)["summary"]
    if rollouts > 1:
        report["groups"], report["group_summary"] = summarize_groups(results)
    return report


def summarize_groups(episodes):
    """Describe return variation (population std) and route diversity without updates."""
    grouped = {}
    for episode in episodes:
        grouped.setdefault(episode["task_index"], []).append(episode)
    groups = []
    for task_index, rows in grouped.items():
        returns = [row["episode_return"] for row in rows]
        groups.append({
            "task_index": task_index, "start": rows[0]["start"], "target": rows[0]["target"],
            "shortest_distance": rows[0]["shortest_distance"], "num_rollouts": len(rows),
            "successes": sum(row["success"] for row in rows), "returns": returns,
            "mean_return": mean(returns), "return_std": pstdev(returns),
            "has_reward_variation": len(set(returns)) > 1,
            "unique_routes": len({tuple(row["path"]) for row in rows}),
            "unique_tool_sequences": len({json.dumps(row["tool_calls"], sort_keys=True) for row in rows}),
        })
    summary = {
        "num_tasks": len(groups), "total_rollouts": len(episodes),
        "groups_with_success": sum(group["successes"] > 0 for group in groups),
        "mixed_success_groups": sum(0 < group["successes"] < group["num_rollouts"] for group in groups),
        "groups_with_reward_variation": sum(group["has_reward_variation"] for group in groups),
        "zero_variance_groups": sum(not group["has_reward_variation"] for group in groups),
        "zero_return_groups": sum(all(value == 0 for value in group["returns"]) for group in groups),
        "average_unique_routes": mean(group["unique_routes"] for group in groups),
    }
    return groups, summary


def main():
    """Evaluate or sample the policy and save compact, audited route records."""
    parser = argparse.ArgumentParser(description="Evaluate or sample the Qwen Wiki Game policy.")
    parser.add_argument("--tasks", type=Path, required=True,
                        help="JSON tasks with start, target, and shortest_distance.")
    parser.add_argument("--adapter", type=Path,
                        help="Local PEFT adapter directory; omit to evaluate the base model.")
    parser.add_argument("--limit", type=int,
                        help="Select up to this many unique tasks using --seed.")
    parser.add_argument("--distance", type=int,
                        help="Keep only tasks with this shortest distance, before applying --limit.")
    parser.add_argument("--sample", action="store_true", help="Sample actions instead of greedy decoding.")
    parser.add_argument("--articles", type=Path, default=Path("data/processed/articles.json"),
                        help="Article data; link_context needs data/processed/articles_context.json.")
    parser.add_argument("--prompt-style", choices=PROMPT_STYLES, default="article_text")
    parser.add_argument("--uncached-greedy", action="store_true",
                        help="Use the original uncached greedy decoder (reproduces the first baselines).")
    parser.add_argument("--rollouts", type=int, default=1, help="Sampled routes per task (default: 1).")
    parser.add_argument("--seed", type=int, default=42, help="Task selection and rollout seed (default: 42).")
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
    if args.distance is not None and args.distance < 1:
        parser.error("--distance must be positive.")

    if args.rollouts < 1 or (args.rollouts > 1 and not args.sample):
        parser.error("Positive --rollouts required; --rollouts > 1 requires --sample.")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1.")

    adapter_hashes = None
    if args.adapter is not None:
        config = args.adapter / "adapter_config.json"
        weights = [args.adapter / name for name in ("adapter_model.safetensors", "adapter_model.bin")
                   if (args.adapter / name).is_file()]
        if not config.is_file() or not weights:
            parser.error("--adapter must contain adapter_config.json and PEFT adapter weights.")
        adapter_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [config, *weights]}

    with args.tasks.open(encoding="utf-8") as file:
        tasks = json.load(file)
    if not isinstance(tasks, list) or not tasks:
        parser.error("The tasks file must contain a non-empty JSON list.")
    if args.distance is not None:
        tasks = [task for task in tasks if task["shortest_distance"] == args.distance]
        if not tasks:
            parser.error(f"No tasks have shortest distance {args.distance}.")
    if args.limit is not None:
        tasks = random.Random(args.seed).sample(tasks, min(args.limit, len(tasks)))

    env = WikiGameEnv(articles_path=args.articles, max_clicks=args.max_clicks)
    try:
        for task in tasks:
            env.validate_pair(task["start"], task["target"])

        if args.uncached_greedy and args.sample:
            parser.error("--uncached-greedy applies to greedy evaluation only.")
        policy = WikiGamePolicy(max_input_tokens=4096, adapter_path=args.adapter,
                                prompt_style=args.prompt_style)
        policy.cached_greedy = not args.uncached_greedy
        runner = EpisodeRunner(env, policy)
        report = evaluate(
            runner, tasks, deterministic=not args.sample, rollouts=args.rollouts, seed=args.seed,
        )
        report["settings"].update({
            "model": policy.model.name_or_path,
            "adapter": str(args.adapter) if args.adapter is not None else None,
            "adapter_sha256": adapter_hashes,
            "tasks_file": str(args.tasks),
            "distance_filter": args.distance,
            "sample_seed": args.seed if args.limit is not None else None,
            "max_input_tokens": policy.max_input_tokens,
            "action_format": "qwen_tool_call",
            "follow_link_argument": "title",
            "enable_thinking": False,
            "use_cache": not args.uncached_greedy,
            "prompt_style": args.prompt_style, "articles": str(args.articles),
            "decoder": ("kv_cached_sampling_then_teacher_forced_rescore" if args.sample
                        else "forced_token_skip" if args.uncached_greedy
                        else "kv_cached_greedy_float32_choices"),
            "system_prompt": policy.system_prompt,
            "observation_template": policy.observation_template,
        })

        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
        print(json.dumps(report["summary"], indent=2))
        if "group_summary" in report:
            print(json.dumps(report["group_summary"], indent=2))
        print(f"Results saved to {output_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
