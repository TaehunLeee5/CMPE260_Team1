"""Standalone Wiki Game policy audit. Run this file separately from the old suite.

    uv run pytest test/llm_policy_audit_test.py -v -s

Apply the production colon protocol edits from COLON_PROTOCOL_README.txt first. Model-quality
misses are XFAIL; invalid decoding or probability mismatches are ordinary FAIL.
"""

from collections import Counter, deque
import copy
import functools
import json
import math
import os
from pathlib import Path
import platform

import pytest
import torch
import transformers

from policy.llm_policy import WikiGamePolicy


ROOT = Path(__file__).resolve().parents[1]
ATOL = float(os.environ.get("WIKI_AUDIT_ATOL", "0.0001"))
DRAWS = int(os.environ.get("WIKI_AUDIT_DRAWS", "128"))
REAL_STATES = int(os.environ.get("WIKI_AUDIT_REAL_STATES", "12"))
INPUT_TOKENS = int(os.environ.get("WIKI_AUDIT_INPUT_TOKENS", "4096"))
if ATOL <= 0 or DRAWS < 1 or REAL_STATES < 1 or INPUT_TOKENS < 1:
    raise pytest.UsageError("WIKI_AUDIT_* settings must be positive.")


def observation(ids=(0, 2, 7)):
    titles = ["Mathematics", "Chemistry", "Astronomy"]
    return {
        "current_title": "Science", "target_title": "Physics",
        "text": "Science includes mathematics, chemistry, astronomy and physics.",
        "history": ["Science"], "remaining_clicks": 20,
        "actions": [{"id": i, "title": titles[n] if len(ids) == 3 else f"Article {i}"}
                    for n, i in enumerate(ids)],
    }


def canonical_sequences(policy, obs):
    return {a["id"]: policy.tokenizer.encode(str(a["id"]), add_special_tokens=False)
            + [policy.action_end_token_id] for a in obs["actions"]}


def assert_matching(old, new, label):
    assert math.isfinite(old) and math.isfinite(new), label
    assert abs(old - new) <= ATOL, (
        f"{label}: rollout={old:.9f}, recomputed={new:.9f}, "
        f"error={abs(old-new):.9f}, ratio={math.exp(new-old):.9f}; "
        "inspect generation settings, precision, model mode and prompt equality."
    )


@pytest.fixture(scope="session")
def audit_report():
    report = {"python": platform.python_version(),
              "torch": torch.__version__, "transformers": transformers.__version__,
              "settings": {"atol": ATOL, "draws": DRAWS,
                           "real_states": REAL_STATES, "input_tokens": INPUT_TOKENS},
              "cases": []}
    yield report
    destination = ROOT / "evaluation/results/policy_audit_colon.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"\nAudit report: {destination}")


@pytest.fixture(scope="session")
def audit_policy(audit_report):
    policy = WikiGamePolicy()
    colon = policy.tokenizer.encode(":", add_special_tokens=False)
    assert len(colon) == 1 and getattr(policy, "action_end_token_id", None) == colon[0], (
        "Apply the colon protocol changes in COLON_PROTOCOL_README.txt before this audit."
    )
    audit_report['action_terminator'] = ':'
    audit_report['action_end_token_id'] = policy.action_end_token_id
    # This test instance has its own bounded context budget; no baseline is edited.
    if policy.max_input_tokens is None:
        policy.max_input_tokens = INPUT_TOKENS
    else:
        policy.max_input_tokens = min(policy.max_input_tokens, INPUT_TOKENS)
    policy.model.eval()
    audit_report.update(model=policy.model.name_or_path,
                        dtype=str(policy.model.dtype), device=str(policy.model.device),
                        audit_input_tokens=policy.max_input_tokens,
                        saved_generation_config=policy.model.generation_config.to_dict())
    print(f"\nAudit runtime: transformers={transformers.__version__}, "
          f"device={policy.model.device}, dtype={policy.model.dtype}")
    return policy


def audit_decision(policy, obs, deterministic, monkeypatch, report, label):
    original_observation = copy.deepcopy(obs)
    captured = {}
    original_generate = policy.model.generate

    @functools.wraps(original_generate)
    def capture(*args, **kwargs):
        result = original_generate(*args, **kwargs)
        captured["length"] = kwargs["input_ids"].shape[1]
        captured["result"] = result
        return result

    with monkeypatch.context() as patch:
        patch.setattr(policy.model, "generate", capture)
        action, old = policy.choose_action(obs, deterministic=deterministic)
    assert obs == original_observation, "Choosing an action mutated its observation."
    sequences = canonical_sequences(policy, obs)
    assert action in sequences, f"Unavailable action: {action}"
    result = captured["result"]
    generated = result.sequences[0, captured["length"]:].tolist()
    assert generated == sequences[action], "Generated tokens do not equal the complete selected ID + action terminator."
    assert len(result.scores) == len(generated)
    # Independent prefix reference, rather than asking _allowed_tokens for the answer.
    for position, scores in enumerate(result.scores):
        prefix = generated[:position]
        expected = {s[position] for s in sequences.values()
                    if len(s) > position and s[:position] == prefix}
        actual = set(torch.where(torch.isfinite(scores[0]))[0].tolist())
        assert actual == expected, (
            f"{label}, prefix={prefix}: generation filtered valid continuations "
            "or admitted invalid ones. Check top-k/top-p and action constraints."
        )
        if deterministic:
            assert generated[position] == scores[0].argmax().item(), (
                f"{label}: deterministic output did not select the highest-scoring next token."
            )
    with torch.no_grad():
        new = policy.get_action_log_prob(obs, action).item()
    record = {"kind": label, "deterministic": deterministic,
              "links": len(sequences), "action": action, "generated_tokens": generated,
              "rollout_log_prob": old, "recomputed_log_prob": new,
              "log_prob_error": abs(old-new), "ratio": math.exp(new-old)}
    report["cases"].append(record)
    assert old <= ATOL
    assert_matching(old, new, label)
    return action, old


def test_exhaustive_prefix_reachability(audit_policy):
    """Every ID at every graph-sized action count is reachable and ends in colon."""
    p = audit_policy
    for count in range(1, 295):
        obs = observation(list(range(count)))
        expected = canonical_sequences(p, obs)
        actual = p._action_sequences(obs["actions"])
        assert actual == expected
        leaves = {tuple(s): a for a, s in expected.items()}
        assert len(leaves) == count
        pending, reached = [[]], set()
        limit = max(map(len, expected.values()))
        while pending:
            prefix = pending.pop()
            if prefix and prefix[-1] == p.action_end_token_id:
                assert tuple(prefix) in leaves
                reached.add(leaves[tuple(prefix)])
                continue
            assert len(prefix) < limit
            choices = p._allowed_tokens(actual, prefix)
            assert choices, f"Unfinishable prefix {prefix} at link count {count}"
            assert len(choices) == len(set(choices))
            pending.extend(prefix + [token] for token in choices)
        assert reached == set(range(count))


@pytest.mark.parametrize("count", [1, 2, 9, 10, 11, 99, 100, 101, 294])
@pytest.mark.parametrize("deterministic", [False, True])
def test_link_count_boundaries(audit_policy, audit_report, monkeypatch, count, deterministic):
    _, old = audit_decision(audit_policy, observation(list(range(count))), deterministic,
                            monkeypatch, audit_report, f"boundary_{count}")
    if count == 1:
        assert old == pytest.approx(0.0, abs=ATOL)


@pytest.mark.parametrize("action", [0, 2, 10, 100])
def test_controlled_rare_actions(audit_policy, audit_report, monkeypatch, action):
    """Force draws, not probabilities, to check all initial digits and endings."""
    obs = observation([0, 2, 10, 100])
    tokens = iter(canonical_sequences(audit_policy, obs)[action])

    def draw(probabilities, num_samples=1, **kwargs):
        token = next(tokens)
        assert probabilities[0, token] > 0, "A valid continuation has zero sampling probability."
        return torch.tensor([[token]], device=probabilities.device)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "multinomial", draw)
        selected, _ = audit_decision(audit_policy, obs, False, patch, audit_report,
                                     f"controlled_{action}")
    assert selected == action


@pytest.mark.parametrize("ids", [(0,), (0, 2, 7), (1, 10, 100), tuple(range(11))])
def test_complete_action_distribution_normalized(audit_policy, ids):
    obs = observation(ids)
    with torch.no_grad():
        scores = torch.stack([audit_policy.get_action_log_prob(obs, a) for a in ids])
    assert torch.isfinite(scores).all().item()
    assert scores.logsumexp(0).item() == pytest.approx(0.0, abs=ATOL)


@pytest.mark.parametrize("deterministic", [False, True])
def test_gradient_enabled_probability_parity(audit_policy, audit_report, deterministic):
    # Single-token IDs keep the full-model gradient graph to one forward pass.
    obs = observation()
    audit_policy.model.eval()
    action, old = audit_policy.choose_action(obs, deterministic=deterministic)
    with torch.enable_grad():
        new = audit_policy.get_action_log_prob(obs, action)
    assert new.requires_grad
    value = new.detach().item()
    audit_report["cases"].append({"kind": "gradient_parity", "action": action,
                                  "deterministic": deterministic,
                                  "error": abs(old-value), "ratio": math.exp(value-old)})
    assert_matching(old, value, "gradient-enabled recomputation")


def test_greedy_repeatability(audit_policy):
    devices = [audit_policy.model.device.index or 0] if audit_policy.model.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        results = []
        for seed in [17, 43, 91]:
            torch.manual_seed(seed)
            results.append(audit_policy.choose_action(observation([0, 2, 10, 100]), True))
    assert len({a for a, _ in results}) == 1, "Greedy action changed with the random seed."
    assert max(lp for _, lp in results) - min(lp for _, lp in results) <= ATOL


def test_sampling_calibration(audit_policy, audit_report):
    obs = observation()
    ids = [a["id"] for a in obs["actions"]]
    with torch.no_grad():
        expected_lp = {a: audit_policy.get_action_log_prob(obs, a).item() for a in ids}
    counts = Counter()
    devices = [audit_policy.model.device.index or 0] if audit_policy.model.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(260)
        for _ in range(DRAWS):
            action, old = audit_policy.choose_action(obs, deterministic=False)
            assert action in expected_lp
            assert_matching(old, expected_lp[action], "repeated sampling")
            counts[action] += 1
    records = []
    for action in ids:
        probability = math.exp(expected_lp[action])
        # Deliberately conservative: statistical smoke test, not exact frequencies.
        allowance = 6 * math.sqrt(DRAWS * probability * max(0.0, 1-probability)) + 3
        records.append({"action": action, "probability": probability,
                        "count": counts[action], "draws": DRAWS})
        assert abs(counts[action] - DRAWS*probability) <= allowance, records
    audit_report["cases"].append({"kind": "sampling_calibration", "distribution": records})
    print(f"\nSampling calibration: {records}")


def test_complete_action_argmax_diagnostic(audit_policy, audit_report, monkeypatch):
    obs = observation([0, 2, 10, 100])
    chosen, _ = audit_decision(audit_policy, obs, True, monkeypatch, audit_report,
                               "greedy_vs_complete_action")
    with torch.no_grad():
        scores = {a['id']: audit_policy.get_action_log_prob(obs, a['id']).item()
                  for a in obs['actions']}
    best = max(scores, key=scores.get)
    result = {"kind": "greedy_definition_diagnostic", "token_greedy": chosen,
              "complete_action_argmax": best, "log_probabilities": scores,
              "different": chosen != best}
    audit_report["cases"].append(result)
    print(f"\nGreedy definition diagnostic (a difference is allowed): {result}")


def test_text_truncation_preserves_actions(audit_policy, monkeypatch):
    obs = observation([0, 2, 10, 100])
    empty = {**obs, "text": ""}
    limit = audit_policy._prepare_inputs(empty, 4)["input_ids"].shape[1] + 32
    obs['text'] = 'Long article content. ' * 1000
    original = copy.deepcopy(obs)
    with monkeypatch.context() as patch:
        patch.setattr(audit_policy, 'max_input_tokens', limit)
        inputs = audit_policy._prepare_inputs(obs, 4)
        assert inputs['input_ids'].shape[1] <= limit
        decoded = audit_policy.tokenizer.decode(inputs['input_ids'][0])
        assert audit_policy.system_prompt in decoded
        assert f"Target article: {obs['target_title']}" in decoded
        for action in obs['actions']:
            assert f"{action['id']}: {action['title']}" in decoded
        patch.setattr(audit_policy, 'max_input_tokens', 1)
        with pytest.raises(ValueError):
            audit_policy._prepare_inputs(obs, 4)
    assert obs == original


@pytest.mark.parametrize('advantage', [1.0, -1.0])
def test_signed_update_on_small_float32_qwen(audit_policy, audit_report, advantage):
    """Test the policy scoring method on a small RANDOM CPU model, not pretrained weights."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    config = Qwen3Config(
        vocab_size=audit_policy.model.config.vocab_size,
        hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=4096, attention_dropout=0.0,
        tie_word_embeddings=True,
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(260)
        small = copy.copy(audit_policy)
        small.model = Qwen3ForCausalLM(config).float().eval()
    for parameter in small.model.parameters():
        parameter.requires_grad_(False)
    weight = small.model.model.norm.weight
    weight.requires_grad_(True)
    with torch.enable_grad():
        before = small.get_action_log_prob(observation(), 0)
        gradient, = torch.autograd.grad(before, weight)
    assert torch.isfinite(gradient).all().item()
    assert gradient.abs().sum().item() > 0
    with torch.no_grad():
        weight.add_(gradient, alpha=0.5 * advantage)
        after = small.get_action_log_prob(observation(), 0).item()
    audit_report['cases'].append({'kind': 'small_random_model_signed_update',
                                  'advantage': advantage, 'before': before.detach().item(),
                                  'after': after})
    assert advantage * (after - before.detach().item()) > 0, (
        'A signed gradient step moved action probability in the wrong direction.'
    )


@pytest.mark.parametrize("target_id", [0, 9, 10, 99, 100, 293])
def test_direct_target_copying_quality(audit_policy, audit_report, monkeypatch, target_id):
    obs = observation(list(range(294)))
    obs.update(current_title="Europe", target_title="France")
    obs["actions"][target_id]["title"] = "France"
    selected, _ = audit_decision(audit_policy, obs, True, monkeypatch, audit_report,
                                 f"direct_target_{target_id}")
    audit_report["cases"].append({"kind": "target_copying", "expected": target_id,
                                  "selected": selected, "correct": selected == target_id})
    if selected != target_id:
        pytest.xfail(f"MODEL QUALITY: target ID {target_id}, selected {selected}; "
                     "decoding/probability checks passed, but target copying failed.")


def shortest_path(articles, start, target):
    queue, parent = deque([start]), {start: None}
    while queue:
        node = queue.popleft()
        if node == target:
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1]
        for dest in articles[node]["outgoing"]:
            if dest not in parent:
                parent[dest] = node
                queue.append(dest)
    return None


@pytest.fixture(scope="session")
def real_observations(audit_report):
    from environment.environment import WikiGameEnv
    article_path = ROOT / "data/processed/articles.json"
    task_path = ROOT / "data/processed/val.json"
    if not article_path.exists() or not task_path.exists():
        pytest.skip("Real-data checks need data/processed/articles.json and val.json.")
    env = WikiGameEnv(article_path, max_clicks=20)
    try:
        tasks = json.loads(task_path.read_text(encoding="utf-8"))
        assert tasks, "Validation task list is empty."
        states = []
        largest = max(env.articles, key=lambda a: len(env.articles[a]["outgoing"]))
        links = env.articles[largest]["outgoing"]
        target = next((a for a in links if a != largest), None)
        if target is not None:
            env.reset(options={"start": largest, "target": target})
            states.append(copy.deepcopy(env.get_llm_observation()))
        # Follow known shortest routes to build real observations without asking
        # the policy to play more games or exposing unvisited text in its prompt.
        for task in tasks:
            path = shortest_path(env.articles, task["start"], task["target"])
            assert path is not None
            assert len(path)-1 == task["shortest_distance"]
            env.reset(options={"start": task["start"], "target": task["target"]})
            for dest in path[1:]:
                if len(states) < REAL_STATES:
                    states.append(copy.deepcopy(env.get_llm_observation()))
                action = env.articles[env.current]["outgoing"].index(dest)
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            if len(states) >= REAL_STATES:
                break
        assert states
        audit_report["real_observation_count"] = len(states)
        return states
    finally:
        env.close()


@pytest.mark.parametrize("index", range(REAL_STATES))
@pytest.mark.parametrize("deterministic", [False, True])
def test_real_validation_probability_parity(audit_policy, audit_report, real_observations,
                                          monkeypatch, index, deterministic):
    if index >= len(real_observations):
        pytest.skip("Not enough distinct collected route positions for this index.")
    audit_decision(audit_policy, real_observations[index], deterministic, monkeypatch,
                   audit_report, f"real_state_{index}")


def test_invalid_actions_rejected(audit_policy):
    obs = observation()
    for action in [-1, 1, 1000]:
        with pytest.raises(ValueError):
            audit_policy.get_action_log_prob(obs, action)
    obs["actions"] = []
    with pytest.raises(ValueError):
        audit_policy.choose_action(obs)


def test_existing_lora_adapters_receive_gradients(audit_policy):
    model = audit_policy.model
    adapters = [(name, p) for name, p in model.named_parameters()
                if p.requires_grad and "lora_" in name]
    if not adapters:
        pytest.skip("LoRA is not attached to WikiGamePolicy yet; rerun after integration.")
    model.eval()
    model.zero_grad(set_to_none=True)
    try:
        with torch.no_grad():
            action = min([0, 2, 7], key=lambda a: audit_policy.get_action_log_prob(observation(), a).item())
        with torch.enable_grad():
            loss = -audit_policy.get_action_log_prob(observation(), action)
            loss.backward()
        gradients = [p.grad for _, p in adapters if p.grad is not None]
        assert gradients, "No trainable LoRA parameter received a gradient."
        assert all(torch.isfinite(g).all().item() for g in gradients)
        assert any(g.abs().sum().item() > 0 for g in gradients)
    finally:
        model.zero_grad(set_to_none=True)


@pytest.mark.parametrize('index', [0, 1])
def test_lora_real_observation_gradient_parity(audit_policy, audit_report, real_observations, index):
    if not getattr(audit_policy.model, 'peft_config', None):
        pytest.skip('LoRA is not attached yet; real-state gradient checks run after integration.')
    if index >= len(real_observations):
        pytest.skip('Not enough real observations.')
    obs = real_observations[index]
    audit_policy.model.eval()
    action, old = audit_policy.choose_action(obs, deterministic=False)
    with torch.enable_grad():
        new = audit_policy.get_action_log_prob(obs, action)
    assert new.requires_grad
    audit_report['cases'].append({'kind': 'lora_real_gradient_parity', 'index': index,
                                  'error': abs(old - new.detach().item())})
    assert_matching(old, new.detach().item(), 'LoRA real-observation gradient parity')


@pytest.mark.parametrize("target_id", [0, 9, 10, 99, 100, 293])
def test_real_title_target_copying(audit_policy, audit_report, real_observations,
                                  monkeypatch, target_id):
    obs = copy.deepcopy(max(real_observations, key=lambda x: len(x["actions"])))
    titles = {a["id"]: a["title"] for a in obs["actions"]}
    if target_id not in titles:
        pytest.skip(f"Largest available article has no action ID {target_id}")
    obs["target_title"] = titles[target_id]
    selected, _ = audit_decision(audit_policy, obs, True, monkeypatch, audit_report,
                                 f"real_target_{target_id}")
    result = {"kind": "real_target_copying", "current_article": obs["current_title"],
              "expected_id": target_id, "expected_title": titles[target_id],
              "selected_id": selected, "selected_title": titles[selected],
              "correct": selected == target_id}
    audit_report["cases"].append(result)
    print(f"\nReal-title copying: {result}")
    if selected != target_id:
        pytest.xfail(f"MODEL QUALITY: expected {target_id}: {titles[target_id]}, "
                     f"selected {selected}: {titles[selected]}")
