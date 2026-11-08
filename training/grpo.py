import argparse
import hashlib
import json
import math
import random
from importlib.metadata import version
from pathlib import Path
from statistics import mean, pstdev

import torch
from tqdm import tqdm
from transformers import set_seed

from environment.environment import WikiGameEnv
from environment.episode_runner import EpisodeRunner
from evaluation.audit_saved_routes import analyze_routes
from evaluation.evaluate import summarize_groups
from policy.llm_policy import WikiGamePolicy


def group_advantages(returns, epsilon=1e-4):
    """Use population std; identical returns always produce exactly zero advantage."""
    if len(returns) < 2 or not all(math.isfinite(value) for value in returns):
        raise ValueError("Each group needs at least two finite returns.")
    average, std = mean(returns), pstdev(returns)
    return [(value - average) / (std + epsilon) for value in returns] if std else [0.0] * len(returns)


def clipped_token_loss(current, old, advantage, clip_range=0.2):
    """Negative clipped surrogate; the recorded behavior probability is constant."""
    ratio = (current - old.detach()).exp()
    return -torch.minimum(ratio * advantage, ratio.clamp(1 - clip_range, 1 + clip_range) * advantage)


def collect_groups(runner, tasks, rollouts=4, seed=42):
    """Seed once, then retain exact prompts, token masks and behavior log probabilities."""
    if not tasks or rollouts < 2:
        raise ValueError("Provide tasks and at least two rollouts per task.")
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


def _branch_scores(policy, episode):
    for step in episode["steps"]:
        scores = policy.iter_token_log_probs(step["prompt_ids"], step["token_ids"], step["token_choices"])
        for current, old, choices in zip(scores, step["token_log_probs"], step["token_choices"]):
            if len(choices) > 1:
                yield current, old


def grpo_update(policy, groups, optimizer, clip_range=0.2, max_grad_norm=1.0, atol=1e-4):
    """Accumulate one exact masked-token update; reject stale/mismatched behavior scores.

    Keep all weights fixed throughout collection and gradient accumulation. Training
    mode activates non-reentrant layer checkpointing; Qwen and adapter dropout must
    both be zero. Rollouts and post-update checks run in evaluation mode.
    """
    if not groups or not 0 < clip_range < 1 or max_grad_norm <= 0:
        raise ValueError("Provide groups, a clip range in (0, 1), and a positive gradient norm limit.")
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
        advantages = group_advantages([e["episode_return"] for e in group])
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
                for current, old in _branch_scores(policy, episode):
                    error = abs(current.detach().item() - old)
                    if not math.isfinite(error) or error > atol:
                        raise RuntimeError("Behavior log probabilities do not match: collect fresh rollouts with unchanged weights and masks.")
                    result["max_initial_token_log_prob_error"] = max(result["max_initial_token_log_prob_error"], error)
                    ratio_error = abs(math.expm1(current.detach().item() - old))
                    result["max_initial_ratio_error"] = max(result["max_initial_ratio_error"], ratio_error)
                    loss = clipped_token_loss(current, current.new_tensor(old), advantage, clip_range) / (count * len(rows))
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite GRPO loss; optimizer was not stepped.")
                    loss.backward()
                    result["loss"] += loss.detach().item()
                    progress.update(1)
        result["initial_probability_checks_passed"] = True
        if any(p.grad is not None for p, _ in frozen):
            raise RuntimeError("A frozen base parameter received a gradient.")
        norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm, error_if_nonfinite=True)
        result["gradient_norm_before_clipping"] = norm.item()
        if norm.item() == 0:
            result["skip_reason"] = "Accumulated adapter gradient is zero."
            return result
        optimizer.step()
        result["optimizer_step_applied"] = True
        result["adapters_changed"] = any(not torch.equal(p.detach(), before) for p, before in zip(trainable, initial))
        result["base_parameter_versions_unchanged"] = all(p._version == before for p, before in frozen)
        if not result["base_parameter_versions_unchanged"] or not all(torch.isfinite(p).all() for p in trainable):
            raise RuntimeError("Frozen weights changed or adapter values became nonfinite.")
        policy.model.eval()
        ratios = []
        with torch.no_grad(), tqdm(total=total_tokens, desc="Checking updated policy", unit="token") as progress:
            for episode, _, _ in active:
                for current, old in _branch_scores(policy, episode):
                    ratio = math.exp(current.item() - old)
                    if not math.isfinite(ratio):
                        raise RuntimeError("A post-update probability ratio is nonfinite.")
                    ratios.append(ratio)
                    progress.update(1)
        result["post_update_token_ratio"] = {
            "mean": mean(ratios), "min": min(ratios), "max": max(ratios),
            "fraction_outside_clip_range": mean(not 1 - clip_range <= r <= 1 + clip_range for r in ratios),
        }
        return result
    finally:
        optimizer.zero_grad(set_to_none=True)
        policy.model.eval()


def compact_report(groups, max_clicks):
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
    return {"settings": {"max_clicks": max_clicks}, "episodes": episodes,
            "groups": group_rows, "group_summary": group_summary}


def main():
    parser = argparse.ArgumentParser(description="Collect one fresh batch and take at most one GRPO LoRA update.")
    parser.add_argument("--tasks", type=Path, default=Path("data/processed/final_5000/train.json"))
    parser.add_argument("--articles", type=Path, default=Path("data/processed/articles.json"))
    parser.add_argument("--distance", type=int, default=2)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-clicks", type=int, default=20)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True, help="A new run directory.")
    args = parser.parse_args()
    if min(args.limit, args.distance, args.max_clicks, args.max_input_tokens) < 1 or args.rollouts < 2:
        parser.error("Budgets, distance and task limit must be positive; rollouts must be at least two.")
    if not 0 <= args.seed < 2**32 or not 0 < args.clip_range < 1:
        parser.error("Invalid seed or clipping range.")
    if not (math.isfinite(args.learning_rate) and args.learning_rate > 0 and
            math.isfinite(args.max_grad_norm) and args.max_grad_norm > 0):
        parser.error("Learning rate and maximum gradient norm must be positive and finite.")
    if args.output.exists():
        parser.error("Output directory already exists; choose a new run directory.")
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    tasks = [task for task in tasks if task["shortest_distance"] == args.distance]
    if not tasks or len({(t["start"], t["target"]) for t in tasks}) != len(tasks):
        parser.error("Provide unique training pairs matching the requested distance.")
    tasks = random.Random(args.seed).sample(tasks, min(args.limit, len(tasks)))
    env = WikiGameEnv(args.articles, max_clicks=args.max_clicks)
    try:
        for task in tasks:
            env.validate_pair(task["start"], task["target"])
        set_seed(args.seed)
        policy = WikiGamePolicy(max_input_tokens=args.max_input_tokens)
        policy.enable_lora()
        policy.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = torch.optim.AdamW((p for p in policy.model.parameters() if p.requires_grad),
                                     lr=args.learning_rate, weight_decay=0.0)
        if policy.model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(policy.model.device)
        groups = collect_groups(EpisodeRunner(env, policy), tasks, args.rollouts, args.seed)
        report = compact_report(groups, args.max_clicks)
        report["route_audit"] = analyze_routes(report, env.articles)["summary"]
        print("Collected groups:", json.dumps(report["group_summary"], indent=2), flush=True)
        report["update"] = grpo_update(policy, groups, optimizer, args.clip_range, args.max_grad_norm)
        report["settings"].update({
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "algorithm": "constrained_token_grpo", "kl_beta": 0.0, "advantage_std_correction": 0,
            "rollout_metrics_phase": "before_optimizer_step",
            "deterministic": False,
            "advantage_epsilon": 1e-4, "loss_normalization": "mean_branch_tokens_per_route_then_mean_all_routes",
            "optimizer_steps_per_batch": 1, "gradient_checkpointing": "non_reentrant",
            "model": policy.model.name_or_path, "device": str(policy.model.device),
            "base_dtype": str(next(policy.model.get_base_model().parameters()).dtype),
            "versions": {name: version(name) for name in ["torch", "transformers", "peft"]},
            "tasks_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
            "system_prompt": policy.system_prompt, "observation_template": policy.observation_template,
            "lora": {"rank": 8, "alpha": 16, "dropout": 0.0, "target_modules": "all-linear"},
            "rollout_mode": "sampled", "temperature": 1.0, "use_cache": False,
            "trainable_parameters": sum(p.numel() for p in policy.model.parameters() if p.requires_grad),
        })
        if policy.model.device.type == "cuda":
            report["peak_allocated_gib"] = torch.cuda.max_memory_allocated(policy.model.device) / 2**30
        args.output.mkdir(parents=True)
        report["adapter_checkpoint"] = None
        if report["update"]["optimizer_step_applied"]:
            policy.model.save_pretrained(args.output / "adapter", save_embedding_layers=False)
            report["adapter_checkpoint"] = str(args.output / "adapter")
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print("GRPO update:", json.dumps(report["update"], indent=2))
        if "peak_allocated_gib" in report:
            print("Peak allocated GiB:", report["peak_allocated_gib"])
        print(f"Saved: {args.output / 'report.json'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
