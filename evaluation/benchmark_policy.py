"""Compare standard uncached generation with forced-token skipping."""

import argparse
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
import transformers
from transformers import GenerationConfig

from environment.environment import WikiGameEnv
from evaluation.audit_saved_routes import analyze_routes
from policy.llm_policy import WikiGamePolicy

ROOT = Path(__file__).resolve().parents[1]


def replay_states(report, env, count):
    """Select states evenly across the saved moves, including the first and last."""
    if count < 1:
        raise ValueError("State count must be positive.")
    analyze_routes(report, env.articles)
    total = sum(episode["clicks"] for episode in report["episodes"])
    count = min(count, total)
    positions = {0} if count == 1 else {i * (total - 1) // (count - 1) for i in range(count)}
    selected, position = [], 0
    for episode in report["episodes"]:
        env.reset(options={"start": episode["start"], "target": episode["target"]})
        for step, call in enumerate(episode["tool_calls"], 1):
            if position in positions:
                selected.append({
                    "start": episode["start"], "target": episode["target"], "step": step,
                    "observation": env.get_llm_observation(),
                })
            action = env.back_action if call["name"] == "go_back" else call["arguments"]["link_id"]
            env.step(action)
            position += 1
    return selected


@torch.no_grad()
def reference_decision(policy, observation):
    """Slow reference: ordinary HF generation over the same complete legal calls."""
    policy.model.eval()
    calls = policy._tool_calls(observation)
    sequences = policy._action_sequences(calls)
    inputs = policy._prepare_inputs(observation, max(map(len, sequences.values())))
    prompt_ids = inputs["input_ids"][0].tolist()
    result = policy.model.generate(
        **inputs,
        generation_config=GenerationConfig(
            max_new_tokens=max(map(len, sequences.values())), do_sample=False,
            eos_token_id=policy.tokenizer.eos_token_id, pad_token_id=policy.tokenizer.eos_token_id,
            return_dict_in_generate=True, output_scores=True, use_cache=False,
        ),
        prefix_allowed_tokens_fn=lambda batch, ids: policy._allowed_tokens(
            sequences, ids[len(prompt_ids):].tolist(),
        ),
        logits_to_keep=1,
    )
    tokens = result.sequences[0, len(prompt_ids):].tolist()
    action = next(a for a, sequence in sequences.items() if sequence == tokens)
    scores = policy.model.compute_transition_scores(
        result.sequences, result.scores, normalize_logits=True,
    )[0]
    return {
        "action": action, "tool_call": calls[action], "prompt_ids": prompt_ids,
        "token_ids": tokens,
        "token_choices": [policy._allowed_tokens(sequences, tokens[:i]) for i in range(len(tokens))],
        "token_log_probs": scores.tolist(), "log_prob": scores.sum().item(),
    }


def benchmark(policy, states, repeats=2, atol=1e-4):
    """Time paired greedy calls; verify probabilities outside the timer.

    Both decoders are uncached. CUDA synchronization includes completed GPU work.
    Modes alternate order and receive identical observations and prompts.
    """
    if not states or repeats < 1 or not math.isfinite(atol) or atol <= 0:
        raise ValueError("Provide states, positive repeats, and a positive finite tolerance.")
    device = policy.model.device

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    decoders = {
        "reference": lambda obs: reference_decision(policy, obs),
        "optimized": lambda obs: policy.choose_action(obs, deterministic=True),
    }
    rows = []
    for decode in decoders.values():
        decode(states[0]["observation"])
        synchronize()
    for state_index, state in enumerate(states):
        for repeat in range(repeats):
            pair = {}
            order = ["reference", "optimized"] if repeat % 2 == 0 else ["optimized", "reference"]
            for mode in order:
                synchronize()
                started = perf_counter()
                decision = decoders[mode](state["observation"])
                synchronize()
                elapsed = perf_counter() - started
                with torch.no_grad():
                    rescored = policy.get_token_log_probs(
                        decision["prompt_ids"], decision["token_ids"], decision["token_choices"],
                    )
                old = torch.tensor(decision["token_log_probs"], device=device)
                token_error = (rescored - old).abs().max().item()
                action_error = abs(rescored.sum().item() - decision["log_prob"])
                rows.append({
                    "state": state_index, "current_title": state["observation"]["current_title"],
                    "target_title": state["observation"]["target_title"],
                    "back_available": state["observation"]["back_available"],
                    "repeat": repeat + 1, "decoder": mode, "seconds": elapsed,
                    "prompt_tokens": len(decision["prompt_ids"]),
                    "generated_tokens": len(decision["token_ids"]),
                    "decision_tokens": sum(len(choices) > 1 for choices in decision["token_choices"]),
                    "tool_call": decision["tool_call"],
                    "max_token_log_prob_error": token_error,
                    "action_log_prob_error": action_error,
                    "unchanged_policy_ratio": math.exp(rescored.sum().item() - decision["log_prob"]),
                    "probability_check_passed": token_error <= atol and action_error <= atol,
                })
                pair[mode] = decision
            if pair["reference"]["prompt_ids"] != pair["optimized"]["prompt_ids"]:
                raise RuntimeError("Decoders received different prompts.")
            same = pair["reference"]["token_ids"] == pair["optimized"]["token_ids"]
            for row in rows[-2:]:
                row["same_greedy_call"] = same
    reference = median(row["seconds"] for row in rows if row["decoder"] == "reference")
    optimized = median(row["seconds"] for row in rows if row["decoder"] == "optimized")
    return {
        "summary": {
            "states": len(states), "repeats": repeats,
            "reference_median_seconds_per_call": reference,
            "optimized_median_seconds_per_call": optimized,
            "speedup": reference / optimized,
            "all_probability_checks_passed": all(row["probability_check_passed"] for row in rows),
            "all_greedy_calls_matched": all(row["same_greedy_call"] for row in rows),
            "max_token_log_prob_error": max(row["max_token_log_prob_error"] for row in rows),
            "max_action_log_prob_error": max(row["action_log_prob_error"] for row in rows),
        },
        "settings": {
            "model": policy.model.name_or_path, "use_cache": False,
            "device": str(device), "dtype": str(policy.model.dtype),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "max_input_tokens": policy.max_input_tokens, "atol": atol,
        },
        "measurements": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--articles", type=Path, default=ROOT / "data/processed/articles.json")
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.states < 1 or args.repeats < 1 or not math.isfinite(args.atol) or args.atol <= 0:
        parser.error("States, repeats, and tolerance must be positive.")
    if args.output.resolve() in {args.results.resolve(), args.articles.resolve()}:
        parser.error("Output must not overwrite either input file.")
    report = json.loads(args.results.read_text(encoding="utf-8"))
    settings = report["settings"]
    env = WikiGameEnv(args.articles, max_clicks=settings["max_clicks"])
    try:
        states = replay_states(report, env, args.states)
    finally:
        env.close()
    actor = WikiGamePolicy(
        model_name=settings["model"], max_input_tokens=settings["max_input_tokens"],
    )
    actor.system_prompt = settings["system_prompt"]
    print(f"Device: {actor.model.device}, dtype: {actor.model.dtype}. "
          f"Benchmarking {len(states)} states, {args.repeats} repeats per mode.", flush=True)
    result = benchmark(actor, states, args.repeats, args.atol)
    result["inputs"] = {"results": str(args.results), "articles": str(args.articles)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"\nSaved: {args.output}")
    if not result["summary"]["all_probability_checks_passed"]:
        parser.exit(1, "Probability parity failed; inspect the report before accepting optimized rollouts.\n")
    if not result["summary"]["all_greedy_calls_matched"]:
        parser.exit(1, "Greedy calls differed; inspect the report before accepting optimized rollouts.\n")


if __name__ == "__main__":
    main()
