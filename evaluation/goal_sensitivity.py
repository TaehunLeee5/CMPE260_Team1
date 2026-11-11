import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch
import transformers

from environment.environment import WikiGameEnv
from evaluation.benchmark_policy import replay_states
from policy.llm_policy import WikiGamePolicy

ROOT = Path(__file__).resolve().parents[1]


# Experimental layout only; production keeps its existing default until evaluated.
ARTICLE_FIRST_TEMPLATE = """Current article text:
{text}

Target article: {target_title}
Current article: {current_title}
Remaining actions: {remaining_clicks}
Visited route: {history}
Navigation stack: {navigation_stack}
Back: {back_destination}

Available links:
{links}"""


def control_targets(observation, count=2):
    """Select spread-out link IDs, excluding targets already present in the route."""
    excluded = set(observation["history"]) | {observation["current_title"], observation["target_title"]}
    eligible = [action for action in observation["actions"] if action["title"] not in excluded]
    count = min(count, len(eligible))
    if count < 1:
        return []
    indices = [0] if count == 1 else [i * (len(eligible) - 1) // (count - 1) for i in range(count)]
    return [eligible[index] for index in indices]


def fixed_context(policy, observation, targets, templates=None):
    """Use one article fragment that fits every requested goal and layout."""
    visible = deepcopy(observation)
    calls = policy._tool_calls(visible)
    output_tokens = max(map(len, policy._action_sequences(calls).values()))
    limit = policy.model.config.max_position_embeddings - output_tokens
    if policy.max_input_tokens is not None:
        limit = min(limit, policy.max_input_tokens)
    text_tokens = policy.tokenizer.encode(visible["text"], add_special_tokens=False)
    original_template = policy.observation_template
    templates = templates or [original_template]
    while True:
        lengths = []
        try:
            for template in templates:
                policy.observation_template = template
                lengths.extend(
                    policy._encode_observation({**visible, "target_title": target})["input_ids"].shape[1]
                    for target in targets
                )
        finally:
            policy.observation_template = original_template
        longest = max(lengths)
        overflow = longest - limit
        if overflow <= 0:
            return visible
        if not text_tokens:
            raise ValueError("Tools, links, and target variants exceed the input token budget.")
        text_tokens = text_tokens[:max(0, len(text_tokens) - overflow)]
        visible["text"] = policy.tokenizer.decode(text_tokens, clean_up_tokenization_spaces=False)


def probe_goals(policy, states, controls=2, *, templates=None):
    """Measure direct-target selection and probabilities without executing new routes."""
    if not states or controls < 1:
        raise ValueError("Provide states and at least one control target.")
    results, skipped = [], []
    for index, state in enumerate(states):
        observation = state["observation"]
        targets = control_targets(observation, controls)
        if not targets:
            skipped.append({"state": index, "reason": "No unvisited alternative linked targets."})
            continue
        variants = [("original", observation["target_title"], None)] + [
            ("direct_link_control", action["title"], action["id"]) for action in targets
        ]
        visible = fixed_context(policy, observation, [title for _, title, _ in variants], templates)
        fixed = {key: value for key, value in visible.items() if key != "target_title"}
        context_hash = hashlib.sha256(json.dumps(fixed, sort_keys=True).encode("utf-8")).hexdigest()
        cases = []
        for kind, title, expected in variants:
            variant = {**deepcopy(visible), "target_title": title}
            decision = policy.choose_action(variant, deterministic=True)
            with torch.no_grad():
                scores = {
                    str(action["id"]): policy.get_action_log_prob(variant, action["id"]).item()
                    for action in targets
                }
            selected_title = (
                variant["back_destination"] if decision["tool_call"]["name"] == "go_back"
                else next(action["title"] for action in variant["actions"] if action["id"] == decision["action"])
            )
            case = {
                "kind": kind, "target_title": title, "expected_action": expected,
                "selected_action": decision["action"], "selected_title": selected_title,
                "tool_call": decision["tool_call"], "selected_log_prob": decision["log_prob"],
                "selected_probability": math.exp(decision["log_prob"]),
                "prompt_tokens": len(decision["prompt_ids"]),
                "control_action_log_probs": scores,
            }
            if expected is not None:
                before = cases[0]["control_action_log_probs"][str(expected)]
                after = scores[str(expected)]
                case.update({
                    "correct": decision["tool_call"]["name"] == "follow_link" and decision["action"] == expected,
                    "action_changed": decision["action"] != cases[0]["selected_action"],
                    "expected_log_prob_before": before, "expected_log_prob_after": after,
                    "expected_log_prob_change": after - before,
                })
            cases.append(case)
        results.append({
            "state": index, "source_start": state["start"], "source_target": state["target"],
            "source_step": state["step"], "current_title": visible["current_title"],
            "remaining_clicks": visible["remaining_clicks"], "history": visible["history"],
            "navigation_stack": visible["navigation_stack"], "back_available": visible["back_available"],
            "fixed_context_sha256": context_hash,
            "article_tokens_retained": len(policy.tokenizer.encode(visible["text"], add_special_tokens=False)),
            "control_targets": targets, "cases": cases,
        })
    comparisons = [case for state in results for case in state["cases"] if case["kind"] == "direct_link_control"]
    correct = sum(case["correct"] for case in comparisons)
    return {
        "summary": {
            "states_probed": len(results), "states_skipped": len(skipped),
            "direct_target_controls": len(comparisons), "direct_target_selections": correct,
            "direct_target_selection_rate_percent": 100 * correct / len(comparisons) if comparisons else None,
            "controls_changing_action": sum(case["action_changed"] for case in comparisons),
            "states_with_any_action_change": sum(
                any(case.get("action_changed", False) for case in state["cases"]) for state in results
            ),
        },
        "states": results, "skipped": skipped,
        "interpretation": {
            "scope": "Synthetic one-step goal interventions, not new navigation episodes or benchmark success rates.",
            "controls": "A control goal is an unvisited current link, reachable in one action. Correct selection requires its exact title.",
            "fixed_context": "Only target_title changes. All variants use identical links, history, budget, Back availability, and article text.",
            "probabilities": "Scores are complete-call log probabilities for the selected control links, keyed by internal action ID, not a full normalized distribution over all actions.",
            "unchanged_action": "An unchanged greedy action alone does not prove goal insensitivity. Inspect correctness and probability shifts; an original action may already match a control.",
            "limits": "Direct-target copying does not establish good intermediate navigation. These probes do not evaluate remote-goal strategy.",
        },
    }



def compare_layouts(policy, states, controls=2):
    """Compare layouts with one text fragment shared across every layout and goal."""
    original = policy.observation_template
    if original == ARTICLE_FIRST_TEMPLATE:
        raise ValueError("The current layout already matches the article-first candidate.")
    templates = {"current": original, "article_first": ARTICLE_FIRST_TEMPLATE}
    layouts = {}
    try:
        for name, template in templates.items():
            policy.observation_template = template
            layouts[name] = probe_goals(policy, states, controls, templates=list(templates.values()))
    finally:
        policy.observation_template = original
    current = layouts["current"]["states"]
    candidate = layouts["article_first"]["states"]
    if len(current) != len(candidate):
        raise RuntimeError("Layout comparison produced different state sets.")
    for before, after in zip(current, candidate):
        if (before["fixed_context_sha256"] != after["fixed_context_sha256"]
                or before["control_targets"] != after["control_targets"]):
            raise RuntimeError("Layout comparison changed context or control targets.")
    return {
        "summary": {name: result["summary"] for name, result in layouts.items()},
        "layouts": layouts,
        "settings": {"layout_templates": templates},
        "interpretation": "Paired one-step controls with identical model, system instructions, tools, state, goals, and article text. Only observation section order changes; production defaults are unchanged.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--articles", type=Path, default=ROOT / "data/processed/articles.json")
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--controls", type=int, default=2)
    parser.add_argument("--compare-layouts", action="store_true",
                        help="Compare current and article-first layouts on identical content.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.states < 1 or args.controls < 1:
        parser.error("States and controls must be positive.")
    if args.output.resolve() in {args.results.resolve(), args.articles.resolve()}:
        parser.error("Output must not overwrite either input file.")
    report = json.loads(args.results.read_text(encoding="utf-8"))
    settings = report["settings"]
    env = WikiGameEnv(args.articles, max_clicks=settings["max_clicks"], self_links=settings.get("self_links", True))
    try:
        states = replay_states(report, env, args.states)
    finally:
        env.close()
    actor = WikiGamePolicy(model_name=settings["model"], max_input_tokens=settings["max_input_tokens"])
    actor.system_prompt = settings["system_prompt"]
    actor.observation_template = settings.get("observation_template", actor.observation_template)
    result = (compare_layouts(actor, states, args.controls) if args.compare_layouts
              else probe_goals(actor, states, args.controls))
    result.setdefault("settings", {}).update({
        **settings, "probe_states": args.states, "controls_per_state": args.controls,
        "device": str(actor.model.device), "dtype": str(actor.model.dtype),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "deterministic": True, "use_cache": False, "decoder": "forced_token_skip",
        "observation_template": actor.observation_template, "compare_layouts": args.compare_layouts,
    })
    result["inputs"] = {"results": str(args.results), "articles": str(args.articles)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
