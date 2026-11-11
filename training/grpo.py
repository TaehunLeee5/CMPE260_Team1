import argparse
import hashlib
import json
import math
import platform
import random
import shutil
from importlib.metadata import version
from pathlib import Path
from collections import Counter
from statistics import mean, pstdev
from time import perf_counter

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import set_seed

from environment.environment import WikiGameEnv
from environment.episode_runner import EpisodeRunner
from evaluation.audit_saved_routes import analyze_routes
from evaluation.evaluate import evaluate, summarize_groups
from policy.llm_policy import PROMPT_STYLES, WikiGamePolicy
from training.checkpoint import (capture_rng, read_checkpoint, resolve_resume, restore_rng,
                                 restore_training_state, save_checkpoint, write_json)


def group_advantages(returns, epsilon=1e-4, min_std=None):
    """Use population std; identical returns always produce exactly zero advantage.

    With min_std, the divisor is max(std, min_std). This keeps groups that differ
    only by the small efficiency bonus (all routes succeed, lengths differ) from
    receiving advantages as large as success versus failure. Without it, the
    original divisor std + epsilon is used.
    """
    if len(returns) < 2 or not all(math.isfinite(value) for value in returns):
        raise ValueError("Each group needs at least two finite returns.")
    if min_std is not None and (not math.isfinite(min_std) or min_std <= 0):
        raise ValueError("min_std must be positive and finite, or None.")
    average, std = mean(returns), pstdev(returns)
    if not std:
        return [0.0] * len(returns)
    divisor = max(std, min_std) if min_std is not None else std + epsilon
    return [(value - average) / divisor for value in returns]


def distance_list(value):
    """Normalize a distance setting; older runs stored a single integer."""
    values = [value] if isinstance(value, int) else list(value)
    if not values or any(type(d) is not int or d < 1 for d in values):
        raise ValueError("Distances must be positive integers.")
    return sorted(set(values))


def sample_batch_tasks(tasks, distances, count, rng=random):
    """Balanced batches: every distance gets an equal share, the remainder goes at random.

    With 4 tasks and distances 2, 3 and 4, each batch has one task at every
    distance plus one at a randomly chosen distance, so each distance still gets
    one third of training on average, but no batch can skip a distance or be
    dominated by one. Tasks within a distance are distinct and chosen uniformly.

    With one distance this is exactly the original random.sample(tasks, count),
    so runs started before multi-distance training resume identically.
    """
    if len(distances) == 1:
        return rng.sample(tasks, count)
    by_distance = {d: [t for t in tasks if t["shortest_distance"] == d] for d in distances}
    rounds, remainder = divmod(count, len(distances))
    per_distance = Counter({d: rounds for d in distances})
    per_distance.update(rng.sample(distances, remainder))
    return [task for d in distances for task in rng.sample(by_distance[d], per_distance[d])]


def clipped_token_loss(current, old, advantage, clip_range=0.2):
    """Negative clipped surrogate; the recorded behavior probability is constant."""
    ratio = (current - old.detach()).exp()
    return -torch.minimum(ratio * advantage, ratio.clamp(1 - clip_range, 1 + clip_range) * advantage)


def collect_groups(runner, tasks, rollouts=4, seed=42):
    """Keep encoded rollouts; seed=None continues the current stream across batches."""
    if not tasks or rollouts < 2:
        raise ValueError("Provide tasks and at least two rollouts per task.")
    if seed is not None:
        set_seed(seed)
    groups = []
    with tqdm(total=len(tasks) * rollouts, desc="Collecting", unit="route") as progress:
        for index, task in enumerate(tasks, 1):
            group = []
            for rollout in range(1, rollouts + 1):
                episode = runner.run_episode(task["start"], task["target"], deterministic=False)
                episode.update(task_index=index, rollout_index=rollout,
                               shortest_distance=task["shortest_distance"])
                # Exact encoded inputs are retained; duplicated article text is unnecessary.
                for step in episode["steps"]:
                    step.pop("observation", None)
                group.append(episode)
                progress.update(1)
            groups.append(group)
    return groups


def _branch_count(episode):
    count = 0
    for step in episode["steps"]:
        if not (len(step["token_ids"]) == len(step["token_choices"]) == len(step["token_log_probs"])):
            raise ValueError("Recorded tokens, masks and probabilities must have equal lengths.")
        for choices, old in zip(step["token_choices"], step["token_log_probs"]):
            if not choices or not math.isfinite(old) or old > 0 or (len(choices) == 1 and old != 0):
                raise ValueError("Invalid recorded behavior probability or token mask.")
            count += len(choices) > 1
    return count


def _error_summary(errors):
    """Distribution of |sampling - recorded| log-probability gaps over decision tokens."""
    if not errors:
        return None
    ordered = sorted(errors)
    return {"decision_tokens": len(ordered), "mean": mean(ordered),
            "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], "max": ordered[-1],
            "fraction_above_0.01": mean(x > 0.01 for x in ordered),
            "fraction_above_0.1": mean(x > 0.1 for x in ordered)}


def _entropy_summary(rows):
    """Behavior-policy entropy at decision tokens: raw (nats) and normalized by log(#choices)."""
    raw, normalized = [], []
    for episode, _, _ in rows:
        for step in episode["steps"]:
            branches = [c for c in step["token_choices"] if len(c) > 1]
            for entropy, choices in zip(step.get("decision_entropies", []), branches):
                raw.append(entropy)
                normalized.append(entropy / math.log(len(choices)))
    if not raw:
        return None
    return {"decision_tokens": len(raw), "mean": mean(raw), "mean_normalized": mean(normalized),
            "fraction_near_deterministic": mean(x < 0.05 for x in raw)}


def validation_subset(all_val, config):
    """Fixed validation pairs for every training seed (val_seed); older runs used the training seed."""
    seed = config.get("val_seed", config["seed"])
    return random.Random(seed).sample(all_val, min(config["val_limit"], len(all_val)))


def prune_checkpoint(path):
    """Drop a superseded checkpoint's adapter and trainer state; keep its small route report."""
    path = Path(path)
    if (path / "trainer_state.pt").is_file():
        (path / "trainer_state.pt").unlink()
    if (path / "adapter").is_dir():
        shutil.rmtree(path / "adapter")


def _step_branch_scores(policy, step):
    """Score one recorded step in a single forward pass; return (current, old) per branch token."""
    positions = [i for i, choices in enumerate(step["token_choices"]) if len(choices) > 1]
    if not positions:
        return []
    scores = policy.get_token_log_probs(step["prompt_ids"], step["token_ids"], step["token_choices"])
    return [(scores[i], step["token_log_probs"][i]) for i in positions]


def _later_epoch(policy, optimizer, active, routes, trainable, frozen, clip_range, max_grad_norm, epoch):
    """One more clipped update on the same batch; the policy has moved since collection.

    The recorded behavior probabilities stay the "old" ones, so ratios now differ
    from 1 and clipping is active. A token counts as clipped when the clamp removes
    its gradient: positive advantage with ratio > 1 + clip, or negative advantage
    with ratio < 1 - clip.
    """
    started = perf_counter()
    stats = {"epoch": epoch, "optimizer_step_applied": False, "loss": 0.0, "branch_tokens": 0,
             "clipped_fraction": 0.0, "ratio_mean": None, "ratio_min": None, "ratio_max": None,
             "gradient_norm_before_clipping": None}
    ratios, clipped = [], 0
    optimizer.zero_grad(set_to_none=True)
    policy.model.train()
    with tqdm(total=sum(count for _, _, count in active), desc=f"Epoch {epoch} gradients", unit="token") as progress:
        for episode, advantage, count in active:
            for step in episode["steps"]:
                branches = _step_branch_scores(policy, step)
                if not branches:
                    continue
                loss = 0
                for current, old in branches:
                    ratio = math.exp(current.detach().item() - old)
                    if not math.isfinite(ratio):
                        raise RuntimeError("A probability ratio became nonfinite; optimizer was not stepped.")
                    ratios.append(ratio)
                    clipped += (advantage > 0 and ratio > 1 + clip_range) or (advantage < 0 and ratio < 1 - clip_range)
                    loss = loss + clipped_token_loss(current, current.new_tensor(old), advantage, clip_range)
                loss = loss / (count * routes)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite GRPO loss; optimizer was not stepped.")
                loss.backward()
                stats["loss"] += loss.detach().item()
                progress.update(len(branches))
    if any(p.grad is not None for p, _ in frozen):
        raise RuntimeError("A frozen base parameter received a gradient.")
    stats.update(branch_tokens=len(ratios), clipped_fraction=clipped / len(ratios),
                 ratio_mean=mean(ratios), ratio_min=min(ratios), ratio_max=max(ratios))
    norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm, error_if_nonfinite=True)
    stats["gradient_norm_before_clipping"] = norm.item()
    if norm.item() > 0:
        optimizer.step()
        stats["optimizer_step_applied"] = True
        if not all(p._version == before for p, before in frozen) or not all(torch.isfinite(p).all() for p in trainable):
            raise RuntimeError("Frozen weights changed or adapter values became nonfinite.")
    else:
        stats["skip_reason"] = "Every active token was clipped, so the gradient is zero."
    optimizer.zero_grad(set_to_none=True)
    stats["seconds"] = perf_counter() - started
    return stats


def grpo_update(policy, groups, optimizer, clip_range=0.2, max_grad_norm=1.0, atol=1e-4,
                *, check_post_update=True, min_advantage_std=None, update_epochs=1):
    """Accumulate exact masked-token updates; reject stale/mismatched behavior scores.

    The first epoch runs on-policy, so every recorded behavior probability must be
    reproduced within atol before the first optimizer step. With update_epochs > 1,
    each later epoch takes one more clipped step on the same batch (see _later_epoch).

    Keep all weights fixed throughout collection and first-epoch gradient
    accumulation. Training mode activates non-reentrant layer checkpointing; Qwen
    and adapter dropout must both be zero. Rollouts and post-update checks run in
    evaluation mode.
    """
    started = perf_counter()
    if not groups or not 0 < clip_range < 1 or max_grad_norm <= 0:
        raise ValueError("Provide groups, a clip range in (0, 1), and a positive gradient norm limit.")
    if type(update_epochs) is not int or update_epochs < 1:
        raise ValueError("update_epochs must be a positive integer.")
    named = [(name, p) for name, p in policy.model.named_parameters() if p.requires_grad]
    if not named or any("lora_" not in name for name, _ in named):
        raise ValueError("Only LoRA adapter parameters may be trainable.")
    trainable = [p for _, p in named]
    if {id(p) for group in optimizer.param_groups for p in group["params"]} != {id(p) for p in trainable}:
        raise ValueError("The optimizer must contain exactly the trainable adapter parameters.")
    if policy.model.config.attention_dropout or any(
        isinstance(module, torch.nn.Dropout) and module.p for module in policy.model.modules()
    ):
        raise ValueError("Disable dropout for rollout/training probability consistency.")

    rows, zero_groups = [], 0
    for group in groups:
        if len({(e["start"], e["target"]) for e in group}) != 1:
            raise ValueError("A reward group must contain routes for the same start-target pair.")
        advantages = group_advantages([e["episode_return"] for e in group], min_std=min_advantage_std)
        zero_groups += all(value == 0 for value in advantages)
        rows.extend((episode, advantage, _branch_count(episode)) for episode, advantage in zip(group, advantages))
    active = [(e, a, count) for e, a, count in rows if a != 0 and count]
    total_tokens = sum(count for _, _, count in active)
    result = {
        "optimizer_step_applied": False, "groups": len(groups), "routes": len(rows),
        "zero_variance_groups": zero_groups, "active_routes": len(active),
        "active_branch_tokens": total_tokens, "loss": 0.0,
        "max_initial_token_log_prob_error": 0.0, "max_initial_ratio_error": 0.0,
        "gradient_norm_before_clipping": None, "adapters_changed": False,
        "initial_probability_checks_passed": None,
        "post_update_token_ratio": None,
        "update_epochs": update_epochs, "optimizer_steps": 0, "later_epochs": [],
        "max_sampling_rescore_error": max(
            (step.get("sampling_max_abs_error", 0.0) for e, _, _ in rows for step in e["steps"]), default=0.0),
        "decision_entropy": _entropy_summary(rows),
        "sampling_rescore_error": _error_summary(
            [x for e, _, _ in rows for step in e["steps"] for x in step.get("sampling_abs_errors", [])]),
        "timings": {"gradient_seconds": 0.0, "post_check_seconds": 0.0},
    }
    optimizer.zero_grad(set_to_none=True)
    if not active:
        result["skip_reason"] = "No nonzero advantages with trainable token choices."
        return result

    frozen = [(p, p._version) for p in policy.model.parameters() if not p.requires_grad]
    initial = [p.detach().clone() for p in trainable]
    try:
        policy.model.train()
        with tqdm(total=total_tokens, desc="Accumulating gradients", unit="token") as progress:
            for episode, advantage, count in active:
                for step in episode["steps"]:
                    branches = _step_branch_scores(policy, step)
                    if not branches:
                        continue
                    loss = 0
                    for current, old in branches:
                        error = abs(current.detach().item() - old)
                        if not math.isfinite(error) or error > atol:
                            raise RuntimeError("Behavior log probabilities do not match: collect fresh rollouts with unchanged weights and masks.")
                        result["max_initial_token_log_prob_error"] = max(result["max_initial_token_log_prob_error"], error)
                        ratio_error = abs(math.expm1(current.detach().item() - old))
                        result["max_initial_ratio_error"] = max(result["max_initial_ratio_error"], ratio_error)
                        loss = loss + clipped_token_loss(current, current.new_tensor(old), advantage, clip_range)
                    # One backward pass per step; each token keeps weight 1 / (count * routes).
                    loss = loss / (count * len(rows))
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite GRPO loss; optimizer was not stepped.")
                    loss.backward()
                    result["loss"] += loss.detach().item()
                    progress.update(len(branches))
        result["initial_probability_checks_passed"] = True
        if any(p.grad is not None for p, _ in frozen):
            raise RuntimeError("A frozen base parameter received a gradient.")
        norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm, error_if_nonfinite=True)
        result["gradient_norm_before_clipping"] = norm.item()
        result["timings"]["gradient_seconds"] = perf_counter() - started
        if norm.item() == 0:
            result["skip_reason"] = "Accumulated adapter gradient is zero."
            return result
        optimizer.step()
        result["optimizer_step_applied"] = True
        result["optimizer_steps"] = 1
        for epoch in range(2, update_epochs + 1):
            stats = _later_epoch(policy, optimizer, active, len(rows), trainable, frozen,
                                 clip_range, max_grad_norm, epoch)
            result["later_epochs"].append(stats)
            if not stats["optimizer_step_applied"]:
                break  # Everything is clipped; further epochs would also be zero.
            result["optimizer_steps"] += 1
        result["adapters_changed"] = any(not torch.equal(p.detach(), before) for p, before in zip(trainable, initial))
        result["base_parameter_versions_unchanged"] = all(p._version == before for p, before in frozen)
        if not result["base_parameter_versions_unchanged"] or not all(torch.isfinite(p).all() for p in trainable):
            raise RuntimeError("Frozen weights changed or adapter values became nonfinite.")
        result["timings"]["gradient_seconds"] = perf_counter() - started
        if not check_post_update:
            return result
        policy.model.eval()
        checking_started = perf_counter()
        ratios = []
        with torch.no_grad(), tqdm(total=total_tokens, desc="Checking updated policy", unit="token") as progress:
            for episode, _, _ in active:
                for step in episode["steps"]:
                    for current, old in _step_branch_scores(policy, step):
                        ratio = math.exp(current.item() - old)
                        if not math.isfinite(ratio):
                            raise RuntimeError("A post-update probability ratio is nonfinite.")
                        ratios.append(ratio)
                        progress.update(1)
        result["post_update_token_ratio"] = {
            "mean": mean(ratios), "min": min(ratios), "max": max(ratios),
            "fraction_outside_clip_range": mean(not 1 - clip_range <= r <= 1 + clip_range for r in ratios),
        }
        result["timings"]["post_check_seconds"] = perf_counter() - checking_started
        return result
    finally:
        optimizer.zero_grad(set_to_none=True)
        policy.model.eval()


def compact_report(groups, max_clicks, reward="success"):
    """Keep audited routes on disk while encoded training inputs stay in memory."""
    episodes = []
    for group in groups:
        for episode in group:
            row = {key: value for key, value in episode.items() if key != "steps"}
            row["tool_calls"] = [step["tool_call"] for step in episode["steps"]]
            row["action_log_probs"] = [step["log_prob"] for step in episode["steps"]]
            row["path_efficiency"] = row["shortest_distance"] / row["clicks"] if row["success"] else None
            episodes.append(row)
    group_rows, group_summary = summarize_groups(episodes)
    return {"settings": {"max_clicks": max_clicks, "reward": reward}, "episodes": episodes,
            "groups": group_rows, "group_summary": group_summary}



DEFAULTS = {
    "tasks": "data/processed/final_v2_5000/train.json",
    "val_tasks": "data/processed/final_v2_5000/val.json",
    "articles": "data/processed/articles.json", "distance": [2, 3, 4],
    "tasks_per_batch": 4, "rollouts": 8, "seed": 42, "max_clicks": 20,
    "max_input_tokens": 4096, "learning_rate": 1e-5, "clip_range": 0.2,
    "max_grad_norm": 1.0, "val_limit": 100, "eval_every": 10, "min_advantage_std": 0.1,
    "update_epochs": 2, "activation_checkpointing": True, "val_seed": 42, "reward": "success",
    "prompt_style": "article_text",
}


def read_tasks(path):
    tasks = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"Expected a non-empty task list: {path}")
    if any(type(t.get("shortest_distance")) is not int or t["shortest_distance"] < 1 for t in tasks):
        raise ValueError("Each task needs a positive integer shortest_distance.")
    if len({(t["start"], t["target"]) for t in tasks}) != len(tasks):
        raise ValueError(f"Duplicate task pairs: {path}")
    return tasks


def file_hash(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def validate(runner, tasks):
    """Keep deterministic validation isolated from the training random stream."""
    rng = capture_rng()
    try:
        return evaluate(runner, tasks, deterministic=True)
    finally:
        restore_rng(rng)


def run_training(runner, optimizer, tasks, val_tasks, output, batches, config, progress=None):
    """Take fresh on-policy batches and checkpoint after every completed batch."""
    output = Path(output)
    policy = runner.policy
    if progress is None:
        progress = {
            "settings": config, "planned_batches": batches, "completed_batches": 0,
            "optimizer_steps": 0, "training_routes": 0, "history": [],
            "validation_tasks": val_tasks, "best_checkpoint": None, "best_validation": None,
            "latest_checkpoint": "checkpoints/batch_000000",
        }
        set_seed(config["seed"])
        save_checkpoint(policy, optimizer, output / progress["latest_checkpoint"], progress)
    progress["planned_batches"] = batches
    write_json(output / "report.json", progress)
    for batch in range(progress["completed_batches"] + 1, batches + 1):
        print(f"Batch {batch}/{batches} | optimizer steps so far: {progress['optimizer_steps']}", flush=True)
        if policy.model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(policy.model.device)
        selected = sample_batch_tasks(tasks, distance_list(config["distance"]), config["tasks_per_batch"])
        collected_at = perf_counter()
        groups = collect_groups(runner, selected, config["rollouts"], seed=None)
        collection_seconds = perf_counter() - collected_at
        report = compact_report(groups, runner.env.max_clicks, runner.env.reward_mode)
        report["route_audit"] = analyze_routes(report, runner.env.articles, runner.env.route_return)["summary"]
        print("Collected groups:", json.dumps(report["group_summary"]), flush=True)
        should_validate = batch % config["eval_every"] == 0 or batch == batches
        updating_at = perf_counter()
        update = grpo_update(policy, groups, optimizer, config["clip_range"], config["max_grad_norm"],
                             check_post_update=should_validate,
                             min_advantage_std=config.get("min_advantage_std"),
                             update_epochs=config.get("update_epochs", 1))
        update_seconds = perf_counter() - updating_at
        del groups
        validation_at = perf_counter()
        validation = validate(runner, val_tasks) if should_validate else None
        validation_seconds = perf_counter() - validation_at if should_validate else 0.0
        if validation:
            print("Validation:", json.dumps(validation["summary"]), flush=True)
        progress["completed_batches"] = batch
        progress["optimizer_steps"] += update.get("optimizer_steps", int(update["optimizer_step_applied"]))
        progress["training_routes"] += len(report["episodes"])
        checkpoint = f"checkpoints/batch_{batch:06d}"
        row = {
            "batch": batch, "optimizer_steps": progress["optimizer_steps"], "checkpoint": checkpoint,
            "tasks": selected, "training": {
                "num_episodes": len(report["episodes"]),
                "successes": sum(e["success"] for e in report["episodes"]),
                "average_return": mean(e["episode_return"] for e in report["episodes"]),
            },
            "group_summary": report["group_summary"], "route_audit": report["route_audit"],
            "update": update, "validation": validation["summary"] if validation else None,
            "timings": {"collection_seconds": collection_seconds, "update_seconds": update_seconds,
                        "validation_seconds": validation_seconds,
                        "compute_seconds": collection_seconds + update_seconds + validation_seconds},
        }
        if policy.model.device.type == "cuda":
            row["peak_allocated_gib"] = torch.cuda.max_memory_allocated(policy.model.device) / 2**30
        if validation:
            summary = validation["summary"]
            previous = progress["best_validation"]
            metric = (summary["successes"], summary["average_return"])
            if previous is None or metric > (previous["successes"], previous["average_return"]):
                progress["best_validation"], progress["best_checkpoint"] = summary, checkpoint
        progress["history"].append(row)
        progress["latest_checkpoint"] = checkpoint
        report.update(update=update, validation=validation, batch=batch, timings=row["timings"])
        save_checkpoint(policy, optimizer, output / checkpoint, progress, report)
        write_json(output / "report.json", progress)
        if config.get("prune_checkpoints", False) and len(progress["history"]) >= 2:
            previous = progress["history"][-2]
            # Keep validated checkpoints (evaluation candidates) and the latest one (resume).
            if previous["validation"] is None:
                prune_checkpoint(output / previous["checkpoint"])
        print("GRPO update:", json.dumps(update, indent=2), flush=True)
        print("Timings:", json.dumps(row["timings"]), flush=True)
        print(f"Saved checkpoint: {output / checkpoint}", flush=True)
    return progress


def main():
    parser = argparse.ArgumentParser(description="Train GRPO in fresh batches with validation and batch-boundary resume.")
    for flag in ["tasks", "val_tasks", "articles"]:
        parser.add_argument("--" + flag.replace("_", "-"), type=Path, help=f"Default: {DEFAULTS[flag]}")
    parser.add_argument("--distance", type=int, nargs="+",
                        help="Training distances; every batch gets an equal share of each. Default: 2 3 4")
    for flag in ["tasks_per_batch", "rollouts", "seed", "max_clicks",
                 "max_input_tokens", "val_limit", "eval_every", "update_epochs", "val_seed"]:
        parser.add_argument("--" + flag.replace("_", "-"), type=int, help=f"Default: {DEFAULTS[flag]}")
    for flag in ["learning_rate", "clip_range", "max_grad_norm", "min_advantage_std"]:
        parser.add_argument("--" + flag.replace("_", "-"), type=float, help=f"Default: {DEFAULTS[flag]}")
    parser.add_argument("--prompt-style", choices=PROMPT_STYLES,
                        help="article_text (default) or link_context (needs --articles "
                             "data/processed/articles_context.json).")
    parser.add_argument("--reward", choices=["success", "progress"],
                        help='Failed routes earn 0 ("success", default) or partial credit for '
                             'ending closer to the target ("progress").')
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction,
                        help="Recompute activations during backward to save memory (default: on). "
                             "Off is faster with identical results but needs much more GPU memory.")
    parser.add_argument("--batches", type=int, help="Total target batches, including completed ones (new run default: 5).")
    parser.add_argument("--output", type=Path, help="A new run directory; inferred on resume.")
    parser.add_argument("--resume", type=Path, help="Run directory or its latest complete checkpoint.")
    args = parser.parse_args()
    supplied = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items() if key in DEFAULTS and value is not None}
    if "distance" in supplied:
        supplied["distance"] = sorted(set(supplied["distance"]))
    payload = None
    try:
        if args.resume:
            output, checkpoint = resolve_resume(args.resume)
            payload = read_checkpoint(checkpoint)
            progress = payload["progress"]
            config = progress["settings"]
            if args.output is not None and args.output.resolve() != output.resolve():
                raise ValueError("Resume uses the existing run directory; do not change --output.")
            def differs(key, value):
                if key in ["tasks", "val_tasks", "articles"]:
                    return Path(value).resolve() != Path(config[key]).resolve()
                if key == "distance":
                    return value != distance_list(config[key])
                return value != config.get(key)
            changed = [key for key, value in supplied.items() if differs(key, value)]
            if changed:
                raise ValueError(f"Cannot change training settings on resume: {', '.join(changed)}")
            batches = args.batches if args.batches is not None else progress["planned_batches"]
            if batches < progress["completed_batches"]:
                raise ValueError("--batches is the total target and cannot precede completed batches.")
        else:
            if args.output is None or args.output.exists():
                raise ValueError("Provide --output pointing to a new run directory.")
            output, progress = args.output, None
            config = {**DEFAULTS, **supplied}
            batches = args.batches if args.batches is not None else 5
        distances = distance_list(config["distance"])
        positive = ["tasks_per_batch", "max_clicks", "max_input_tokens", "val_limit", "eval_every"]
        if config.get("update_epochs", 1) < 1:
            raise ValueError("--update-epochs must be at least 1.")
        if batches < 1 or any(config[key] < 1 for key in positive) or config["rollouts"] < 2:
            raise ValueError("Batch counts and budgets must be positive; rollouts must be at least two.")
        if not 0 <= config.get("val_seed", 0) < 2**32:
            raise ValueError("--val-seed must be between 0 and 2**32 - 1.")
        if not 0 <= config["seed"] < 2**32 or not 0 < config["clip_range"] < 1:
            raise ValueError("Invalid seed or clipping range.")
        if any(not math.isfinite(config[key]) or config[key] <= 0 for key in ["learning_rate", "max_grad_norm"]):
            raise ValueError("Learning rate and maximum gradient norm must be positive and finite.")
        floor = config.get("min_advantage_std")
        if floor is not None and (not math.isfinite(floor) or floor <= 0):
            raise ValueError("--min-advantage-std must be positive and finite.")
        all_tasks, all_val = read_tasks(config["tasks"]), read_tasks(config["val_tasks"])
        if {(t["start"], t["target"]) for t in all_tasks} & {(t["start"], t["target"]) for t in all_val}:
            raise ValueError("Training and validation task pairs must be disjoint.")
        tasks = [t for t in all_tasks if t["shortest_distance"] in distances]
        short = [d for d in distances
                 if sum(t["shortest_distance"] == d for t in tasks) < config["tasks_per_batch"]]
        if short:
            raise ValueError(f"Fewer than --tasks-per-batch training pairs at distance(s): {short}")
        val_tasks = validation_subset(all_val, config)
        hashes = {key: file_hash(config[key]) for key in ["tasks", "val_tasks", "articles"]}
        if payload and (hashes != config["data_sha256"] or val_tasks != progress["validation_tasks"]):
            raise ValueError("Training/validation/article data changed since the checkpoint.")
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    if payload and batches == progress["completed_batches"]:
        print(f"Already completed {batches} batches. Use a larger --batches target to continue.")
        return

    try:
        env = WikiGameEnv(config["articles"], max_clicks=config["max_clicks"],
                          reward=config.get("reward", "success"))
    except (ValueError, OSError) as error:
        parser.error(f"Could not create the environment: {error}")
    try:
        for task in tasks + val_tasks:
            env.validate_pair(task["start"], task["target"])
            # With progress rewards, the matrix must agree with the task splits.
            if env.reward_mode == "progress" and \
                    env.shortest_distance(task["start"], task["target"]) != task["shortest_distance"]:
                raise ValueError("The distance matrix disagrees with a task's shortest_distance.")
        set_seed(config["seed"])
        policy = WikiGamePolicy(max_input_tokens=config["max_input_tokens"],
                                prompt_style=config.get("prompt_style", "article_text"))
        if payload:
            policy.model = PeftModel.from_pretrained(policy.model, checkpoint / "adapter",
                                                     is_trainable=True, local_files_only=True)
            policy.model.eval()
        else:
            policy.enable_lora()
        if config.get("activation_checkpointing", True):
            policy.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            # Without checkpointing the KV cache works in training mode, so training can
            # replay the exact sampling computation and rollouts skip the rescoring pass.
            policy.replay_scoring = True
        optimizer = torch.optim.AdamW((p for p in policy.model.parameters() if p.requires_grad),
                                     lr=config["learning_rate"], weight_decay=0.0)
        runtime = {"model": policy.model.name_or_path, "device": str(policy.model.device),
                   "base_dtype": str(next(policy.model.get_base_model().parameters()).dtype),
                   "python": platform.python_version(),
                   "gpu_name": torch.cuda.get_device_name(policy.model.device) if policy.model.device.type == "cuda" else None,
                   "versions": {name: version(name) for name in ["torch", "transformers", "peft", "numpy"]}}
        if payload:
            if runtime != config["runtime"]:
                raise ValueError("Resume requires the same model, device, dtype and package versions.")
            if policy.system_prompt != config["system_prompt"] or policy.observation_template != config["observation_template"]:
                raise ValueError("Policy prompts changed since the checkpoint.")
            restore_training_state(policy, optimizer, payload)
            del payload
        else:
            config.update(
                algorithm="constrained_token_grpo", kl_beta=0.0, advantage_std_correction=0,
                advantage_epsilon=None, advantage_divisor="max(population_std, min_advantage_std)",
                distance_sampling="balanced_equal_share_per_batch_random_remainder",
                rollout_decoder=("kv_cached_sampling" if policy.replay_scoring
                                 else "kv_cached_sampling_then_teacher_forced_rescore"),
                choice_logits="float32_output_layer_over_legal_tokens",
                validation_decoder="kv_cached_greedy_float32_choices",
                prune_checkpoints=True,
                progress_reward=("0.25 * (d_start - d_closest) / d_start, paid per new closest approach"
                                 if config["reward"] == "progress" else None),
                training_scorer=("replay_of_cached_sampling_computation" if policy.replay_scoring
                                 else "one_teacher_forced_pass_per_step"),
                loss_normalization="mean_branch_tokens_per_route_then_mean_all_routes",
                optimizer_steps_per_batch=config["update_epochs"],
                gradient_checkpointing="non_reentrant" if config["activation_checkpointing"] else "off",
                post_update_checks="validation_batches", data_sha256=hashes, runtime=runtime,
                training_metrics_phase="before_optimizer_step", validation_metrics_phase="after_optimizer_step",
                system_prompt=policy.system_prompt, observation_template=policy.observation_template,
                lora={"rank": 8, "alpha": 16, "dropout": 0.0, "target_modules": "all-linear"},
                temperature=1.0, use_cache=False, follow_link_argument="title",
            )
            output.mkdir(parents=True)
        summary = run_training(EpisodeRunner(env, policy), optimizer, tasks, val_tasks, output, batches, config, progress)
        print(f"Completed batches: {summary['completed_batches']} | optimizer steps: {summary['optimizer_steps']}")
        print(f"Best validation checkpoint: {summary['best_checkpoint']}")
        print(f"Saved: {output / 'report.json'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
