from copy import deepcopy
from collections import Counter
import json
import math
import string
import sys

import pytest
import torch
from peft import PeftModel
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from environment.environment import WikiGameEnv
from environment.episode_runner import EpisodeRunner
from policy.llm_policy import WikiGamePolicy
from training.grpo import clipped_token_loss, group_advantages, grpo_update


@pytest.fixture(scope="module")
def policy():
    """Use a local character tokenizer and random Qwen architecture, not trained weights."""
    previous_threads = torch.get_num_threads()
    previous_rng = torch.random.get_rng_state()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    vocabulary = {"<|im_end|>": 0, "<|pad|>": 1, "<|unk|>": 2}
    vocabulary.update({char: index + 3 for index, char in enumerate(string.printable)})
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<|unk|>"))
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<|im_end|>",
        pad_token="<|pad|>", unk_token="<|unk|>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}"
        "{% if loop.first %}\n<tools>{{ tools | tojson }}</tools>{% endif %}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n"
        "{% if not enable_thinking %}<think>\n\n</think>\n\n{% endif %}{% endif %}"
    )
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=4096, attention_dropout=0.0,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    ))
    yield WikiGamePolicy(model=model, tokenizer=tokenizer)
    torch.set_num_threads(previous_threads)
    torch.random.set_rng_state(previous_rng)


@pytest.fixture
def observation():
    return {
        "current_title": "Europe", "target_title": "France",
        "text": "Europe includes many countries.", "history": ["Earth", "Europe"],
        "navigation_stack": ["Earth", "Europe"], "remaining_clicks": 10,
        "back_available": True, "back_destination": "Earth", "back_action_id": 300,
        "actions": [{"id": i, "title": title} for i, title in [
            (0, "Germany"), (2, "Italy"), (10, "Spain"), (100, "France"),
        ]],
    }


def assert_decision(policy, observation, decision, atol=1e-5):
    calls = policy._tool_calls(observation)
    assert decision["tool_call"] == calls[decision["action"]]
    assert json.loads(json.dumps(decision)) == decision
    tokens = decision["token_ids"]
    assert tokens[-1] == policy.tokenizer.eos_token_id
    raw = policy.tokenizer.decode(tokens[:-1], clean_up_tokenization_spaces=False)
    assert raw.startswith("<tool_call>\n") and raw.endswith("\n</tool_call>")
    assert json.loads(raw[len("<tool_call>\n"):-len("\n</tool_call>")]) == decision["tool_call"]
    with torch.enable_grad():
        rescored = policy.get_token_log_probs(
            decision["prompt_ids"], tokens, decision["token_choices"],
        )
    expected = torch.tensor(decision["token_log_probs"], device=rescored.device)
    assert rescored.requires_grad
    assert torch.allclose(rescored.detach(), expected, atol=atol, rtol=0)
    assert math.isfinite(decision["log_prob"]) and decision["log_prob"] <= 0
    assert rescored.detach().sum().item() == pytest.approx(decision["log_prob"], abs=atol)
    assert math.exp(rescored.detach().sum().item() - decision["log_prob"]) == pytest.approx(1, abs=atol)
    for score, choices in zip(decision["token_log_probs"], decision["token_choices"]):
        if len(choices) == 1:
            assert score == 0


def test_prompt_and_tool_definitions(policy, observation):
    original = deepcopy(observation)
    inputs = policy._prepare_inputs(observation, 80)
    prompt = policy.tokenizer.decode(inputs["input_ids"][0])
    for text in ["follow_link", "go_back", "link_id", "Target article: France",
                 "Back: Earth", "Navigation stack: Earth -> Europe", "<think>\n\n</think>"]:
        assert text in prompt
    assert "back_action_id" not in prompt
    assert observation == original
    observation.update(back_available=False, back_action_id=None, back_destination=None)
    prompt = policy.tokenizer.decode(policy._prepare_inputs(observation, 80)["input_ids"][0])
    assert '"name": "go_back"' not in prompt
    assert "Back: unavailable" in prompt
    assert set(policy._tool_calls(observation)) == {0, 2, 10, 100}


@pytest.mark.parametrize("count", [1, 2, 9, 10, 11, 99, 100, 101, 294])
def test_all_calls_are_reachable(policy, observation, count):
    observation["actions"] = [{"id": i, "title": str(i)} for i in range(count)]
    calls = policy._tool_calls(observation)
    sequences = policy._action_sequences(calls)
    assert len({tuple(s) for s in sequences.values()}) == count + 1
    for action, tokens in sequences.items():
        for position, token in enumerate(tokens):
            assert token in policy._allowed_tokens(sequences, tokens[:position])
        assert not policy._allowed_tokens(sequences, tokens)
        text = policy.tokenizer.decode(tokens[:-1])
        assert json.loads(text[len("<tool_call>\n"):-len("\n</tool_call>")]) == calls[action]


@pytest.mark.parametrize("deterministic", [False, True])
def test_rollout_and_training_probabilities_match(policy, observation, deterministic):
    decision = policy.choose_action(observation, deterministic)
    assert_decision(policy, observation, decision)
    with torch.no_grad():
        total = policy.get_action_log_prob(observation, decision["action"]).item()
    assert total == pytest.approx(decision["log_prob"], abs=1e-5)


def test_model_sampling_defaults_do_not_change_policy(policy, observation, monkeypatch):
    defaults = deepcopy(policy.model.generation_config)
    defaults.temperature, defaults.top_p, defaults.top_k = 0.6, 0.95, 20
    monkeypatch.setattr(policy.model, "generation_config", defaults)
    assert_decision(policy, observation, policy.choose_action(observation))


@pytest.mark.parametrize("action", [0, 10, 100, 300])
def test_controlled_draw_covers_both_tools(policy, observation, monkeypatch, action):
    force_action(policy, observation, action, monkeypatch)
    decision = policy.choose_action(observation)
    assert decision["action"] == action
    assert_decision(policy, observation, decision)


def test_complete_action_distribution_normalized(policy, observation):
    with torch.no_grad():
        scores = torch.stack([
            policy.get_action_log_prob(observation, action)
            for action in policy._tool_calls(observation)
        ])
    assert scores.logsumexp(dim=0).item() == pytest.approx(0, abs=1e-5)


@pytest.mark.parametrize("back_only", [False, True])
def test_single_legal_call_has_probability_one(policy, observation, back_only):
    if back_only:
        observation["actions"] = []
        expected = 300
    else:
        observation.update(back_available=False, back_action_id=None, back_destination=None)
        observation["actions"] = [{"id": 100, "title": "France"}]
        expected = 100
    forwards = []
    hook = policy.model.register_forward_pre_hook(lambda module, inputs: forwards.append(1))
    try:
        decision = policy.choose_action(observation)
    finally:
        hook.remove()
    assert forwards == []
    assert decision["action"] == expected
    assert decision["log_prob"] == 0
    assert_decision(policy, observation, decision)


@pytest.mark.parametrize("advantage", [1.0, -1.0])
def test_signed_policy_update(policy, observation, advantage):
    model = deepcopy(policy.model)
    actor = WikiGamePolicy(model=model, tokenizer=policy.tokenizer)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = actor.get_action_log_prob(observation, 0)
    assert before.requires_grad
    (-advantage * before).backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
    optimizer.step()
    with torch.no_grad():
        after = actor.get_action_log_prob(observation, 0).item()
    assert advantage * (after - before.detach().item()) > 0


def test_text_truncation_preserves_state_and_links(policy, observation, monkeypatch):
    empty = {**observation, "text": ""}
    limit = policy._prepare_inputs(empty, 80)["input_ids"].shape[1] + 20
    monkeypatch.setattr(policy, "max_input_tokens", limit)
    observation["text"] = "Article text. " * 1000
    original = deepcopy(observation)
    inputs = policy._prepare_inputs(observation, 80)
    prompt = policy.tokenizer.decode(inputs["input_ids"][0])
    assert inputs["input_ids"].shape[1] <= limit
    for action in observation["actions"]:
        assert f"{action['id']}: {action['title']}" in prompt
    assert "Back: Earth" in prompt and "Visited route: Earth -> Europe" in prompt
    assert observation == original
    monkeypatch.setattr(policy, "max_input_tokens", 1)
    with pytest.raises(ValueError, match="state, and links exceed"):
        policy._prepare_inputs(observation, 80)


def test_invalid_actions_and_records_rejected(policy, observation):
    with pytest.raises(ValueError, match="not available"):
        policy.get_action_log_prob(observation, 99)
    with pytest.raises(ValueError, match="recorded token"):
        policy.get_token_log_probs([1], [1], [[2]])
    observation["actions"].append(observation["actions"][0])
    with pytest.raises(ValueError, match="unique"):
        policy.choose_action(observation)
    observation["actions"] = []
    observation["back_available"] = False
    with pytest.raises(ValueError, match="no available actions"):
        policy.choose_action(observation)


def test_real_policy_runner_roundtrip(policy, tmp_path):
    articles = {
        "Start": {"title": "Start", "text": "Start", "outgoing": ["Dead"]},
        "Dead": {"title": "Dead", "text": "Dead", "outgoing": []},
        "Goal": {"title": "Goal", "text": "PRIVATE", "outgoing": []},
    }
    path = tmp_path / "articles.json"
    path.write_text(json.dumps(articles), encoding="utf-8")
    env = WikiGameEnv(path, max_clicks=2)
    episode = EpisodeRunner(env, policy).run_episode("Start", "Goal")
    assert episode["path"] == ["Start", "Dead", "Start"]
    assert [step["tool_call"]["name"] for step in episode["steps"]] == ["follow_link", "go_back"]
    assert json.loads(json.dumps(episode)) == episode
    for step in episode["steps"]:
        assert_decision(policy, step["observation"], step)


@pytest.fixture(scope="module")
def real_policy():
    actor = WikiGamePolicy()
    # Probability parity needs gradients, but not a full fine-tuning graph.
    for parameter in actor.model.parameters():
        parameter.requires_grad_(False)
    actor.model.model.norm.weight.requires_grad_(True)
    return actor


@pytest.mark.integration
@pytest.mark.parametrize("deterministic", [False, True])
def test_pretrained_rollout_and_training_match(real_policy, observation, deterministic):
    decision = real_policy.choose_action(observation, deterministic)
    assert_decision(real_policy, observation, decision, atol=1e-4)
    print("Tool:", decision["tool_call"], "log probability:", decision["log_prob"])


@pytest.mark.integration
@pytest.mark.parametrize("action", [10, 100, 300])
@pytest.mark.parametrize("long_context", [False, True])
def test_pretrained_controlled_calls(real_policy, observation, monkeypatch, action, long_context):
    if long_context:
        observation["text"] = "Article context. " * 2000
    force_action(real_policy, observation, action, monkeypatch)
    decision = real_policy.choose_action(observation)
    assert decision["action"] == action
    if long_context:
        assert len(decision["prompt_ids"]) > 3000
    assert_decision(real_policy, observation, decision, atol=1e-4)



def force_action(policy, observation, action, monkeypatch):
    """Control draws at branching prefixes without changing their probabilities."""
    sequences = policy._action_sequences(policy._tool_calls(observation))
    tokens = sequences[action]
    branches = []
    for position, token in enumerate(tokens):
        choices = policy._allowed_tokens(sequences, tokens[:position])
        if len(choices) > 1:
            branches.append((choices.index(token), len(choices)))
    draws = iter(branches)

    def draw(probabilities, num_samples=1, **kwargs):
        index, size = next(draws)
        assert probabilities.ndim == 1 and probabilities.numel() == size
        assert probabilities[index] > 0
        return torch.tensor([index], device=probabilities.device)

    monkeypatch.setattr(torch, "multinomial", draw)


def test_forward_passes_only_at_branching_prefixes(policy, observation):
    forwards = []
    hook = policy.model.register_forward_pre_hook(lambda module, inputs: forwards.append(1))
    try:
        decision = policy.choose_action(observation, deterministic=True)
    finally:
        hook.remove()
    decisions = sum(len(choices) > 1 for choices in decision["token_choices"])
    assert len(forwards) == decisions
    assert decisions < len(decision["token_ids"])


def test_standard_uncached_reference_agrees(policy, observation):
    from evaluation.benchmark_policy import reference_decision

    reference = reference_decision(policy, observation)
    optimized = policy.choose_action(observation, deterministic=True)
    assert reference["prompt_ids"] == optimized["prompt_ids"]
    assert reference["token_ids"] == optimized["token_ids"]
    assert reference["token_log_probs"] == pytest.approx(optimized["token_log_probs"], abs=1e-5)


def test_benchmark_pairs_reference_and_optimized(policy, observation):
    from evaluation.benchmark_policy import benchmark

    result = benchmark(policy, [{"observation": observation}], repeats=1, atol=1e-5)
    assert {row["decoder"] for row in result["measurements"]} == {"reference", "optimized"}
    assert result["summary"]["all_probability_checks_passed"]
    assert result["summary"]["all_greedy_calls_matched"]
    assert result["summary"]["speedup"] > 0


def test_sampled_actions_match_complete_call_probabilities(policy, observation):
    with torch.no_grad():
        expected = {
            action: policy.get_action_log_prob(observation, action).exp().item()
            for action in policy._tool_calls(observation)
        }
    draws = 128
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        counts = Counter(policy.choose_action(observation)["action"] for _ in range(draws))
    for action, probability in expected.items():
        tolerance = 6 * math.sqrt(probability * (1 - probability) / draws) + 2 / draws
        assert abs(counts[action] / draws - probability) <= tolerance


def test_goal_controls_exclude_visited_and_original_targets(observation):
    from evaluation.goal_sensitivity import control_targets

    assert [a["id"] for a in control_targets(observation)] == [0, 10]
    assert [a["id"] for a in control_targets(observation, 1)] == [0]
    observation["history"].append("Germany")
    assert [a["id"] for a in control_targets(observation)] == [2, 10]


def test_goal_variants_share_identical_truncated_text(policy, observation, monkeypatch):
    from evaluation.goal_sensitivity import fixed_context

    targets = ["France", "A much longer target title " * 4]
    empty = {**observation, "text": "", "target_title": targets[1]}
    limit = policy._encode_observation(empty)["input_ids"].shape[1] + 30
    monkeypatch.setattr(policy, "max_input_tokens", limit)
    observation["text"] = "Article context. " * 1000
    original = deepcopy(observation)
    visible = fixed_context(policy, observation, targets)
    assert visible["text"] != observation["text"]
    output_tokens = max(map(len, policy._action_sequences(policy._tool_calls(visible)).values()))
    for title in targets:
        variant = {**visible, "target_title": title}
        encoded = policy._encode_observation(variant)["input_ids"]
        prepared = policy._prepare_inputs(variant, output_tokens)["input_ids"]
        assert encoded.shape[1] <= limit
        assert torch.equal(encoded, prepared.cpu())  # No extra target-dependent text truncation.
    assert observation == original


def test_goal_probe_changes_only_target_and_keeps_original_state(policy, observation, monkeypatch):
    from evaluation.goal_sensitivity import probe_goals

    original = deepcopy(observation)
    received = []
    choose = policy.choose_action

    def record(variant, deterministic=False):
        received.append(deepcopy(variant))
        return choose(variant, deterministic)

    monkeypatch.setattr(policy, "choose_action", record)
    result = probe_goals(policy, [{"start": "Earth", "target": "France", "step": 2, "observation": observation}])
    assert len(received) == 3
    contexts = [{k: v for k, v in obs.items() if k != "target_title"} for obs in received]
    assert contexts[0] == contexts[1] == contexts[2]
    assert observation == original
    assert result["summary"]["direct_target_controls"] == 2
    assert result["summary"]["states_probed"] == 1
    assert json.loads(json.dumps(result)) == result


def test_goal_probe_reports_probability_changes_without_argmax_change(policy, observation, monkeypatch):
    from evaluation.goal_sensitivity import probe_goals

    def probability(variant, action):
        # The action distribution changes with the target, but action 0 stays largest.
        return {
            "France": {0: 0.8, 10: 0.01},
            "Germany": {0: 0.9, 10: 0.01},
            "Spain": {0: 0.6, 10: 0.3},
        }[variant["target_title"]][action]

    def choose(variant, deterministic=False):
        return {
            "action": 0, "tool_call": {"name": "follow_link", "arguments": {"link_id": 0}},
            "log_prob": math.log(probability(variant, 0)), "prompt_ids": [1],
        }

    monkeypatch.setattr(policy, "choose_action", choose)
    monkeypatch.setattr(policy, "get_action_log_prob", lambda variant, action: torch.tensor(math.log(probability(variant, action))))
    result = probe_goals(policy, [{"start": "Earth", "target": "France", "step": 2, "observation": observation}])
    assert result["summary"]["controls_changing_action"] == 0
    assert result["summary"]["direct_target_selections"] == 1
    assert result["states"][0]["cases"][2]["expected_log_prob_change"] > 3


def test_goal_probe_skips_states_without_link_controls(policy, observation, monkeypatch):
    from evaluation.goal_sensitivity import probe_goals

    observation["actions"] = []  # Only Back is legal, so changing the goal cannot change the action.

    def unexpected_call(*args, **kwargs):
        raise AssertionError("An uninformative state should not call the model.")

    monkeypatch.setattr(policy, "choose_action", unexpected_call)
    result = probe_goals(policy, [{"start": "Earth", "target": "France", "step": 2, "observation": observation}])
    assert result["summary"]["states_skipped"] == 1
    assert result["summary"]["direct_target_controls"] == 0
    assert result["summary"]["direct_target_selection_rate_percent"] is None


def test_article_first_layout_changes_order_only(policy, observation, monkeypatch):
    from evaluation.goal_sensitivity import ARTICLE_FIRST_TEMPLATE

    current = policy.build_prompt(observation)
    monkeypatch.setattr(policy, "observation_template", ARTICLE_FIRST_TEMPLATE)
    candidate = policy.build_prompt(observation)
    assert current.index("Target article:") < current.index("Current article text:")
    assert candidate.index("Current article text:") < candidate.index("Target article:") < candidate.index("Available links:")
    for value in [observation["text"], observation["target_title"], "0: Germany", "100: France", "Earth -> Europe"]:
        assert value in current and value in candidate
    assert_decision(policy, observation, policy.choose_action(observation, deterministic=True))


def test_joint_text_fits_both_layouts_and_goals(policy, observation, monkeypatch):
    from evaluation.goal_sensitivity import ARTICLE_FIRST_TEMPLATE, fixed_context

    original_template = policy.observation_template
    templates = [original_template, ARTICLE_FIRST_TEMPLATE]
    targets = ["France", "A substantially longer goal title " * 3]
    empty = {**observation, "text": "", "target_title": targets[1]}
    limit = policy._encode_observation(empty)["input_ids"].shape[1] + 30
    monkeypatch.setattr(policy, "max_input_tokens", limit)
    observation["text"] = "Article context. " * 1000
    visible = fixed_context(policy, observation, targets, templates)
    assert policy.observation_template == original_template
    output_tokens = max(map(len, policy._action_sequences(policy._tool_calls(visible)).values()))
    try:
        for template in templates:
            policy.observation_template = template
            for target in targets:
                variant = {**visible, "target_title": target}
                raw = policy._encode_observation(variant)["input_ids"]
                prepared = policy._prepare_inputs(variant, output_tokens)["input_ids"]
                assert raw.shape[1] <= limit
                assert torch.equal(raw, prepared.cpu())
    finally:
        policy.observation_template = original_template


def test_layout_comparison_pairs_states_and_restores_default(policy, observation):
    from evaluation.goal_sensitivity import compare_layouts

    original = policy.observation_template
    state = {"start": "Earth", "target": "France", "step": 2, "observation": observation}
    result = compare_layouts(policy, [state])
    assert policy.observation_template == original
    assert set(result["summary"]) == {"current", "article_first"}
    before = result["layouts"]["current"]["states"][0]
    after = result["layouts"]["article_first"]["states"][0]
    assert before["fixed_context_sha256"] == after["fixed_context_sha256"]
    assert before["control_targets"] == after["control_targets"]
    assert [case["target_title"] for case in before["cases"]] == [case["target_title"] for case in after["cases"]]
    assert json.loads(json.dumps(result)) == result


def test_layout_comparison_restores_default_after_failure(policy, observation, monkeypatch):
    from evaluation.goal_sensitivity import compare_layouts

    original = policy.observation_template

    def fail(*args, **kwargs):
        raise RuntimeError("Simulated model error")

    monkeypatch.setattr(policy, "choose_action", fail)
    with pytest.raises(RuntimeError, match="Simulated model error"):
        compare_layouts(policy, [{"start": "Earth", "target": "France", "step": 2, "observation": observation}])
    assert policy.observation_template == original


def check_lora_update(actor, observation, decision, *, atol):
    """Check training plumbing with one controlled likelihood step, not an RL update."""
    actor.enable_lora()
    assert_decision(actor, observation, decision, atol=atol)
    config = actor.model.peft_config["default"]
    assert config.r == 8 and config.lora_alpha == 16 and config.lora_dropout == 0
    trainable = {name: p for name, p in actor.model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in name for name in trainable)
    frozen = [(p, p._version) for p in actor.model.parameters() if not p.requires_grad]
    initial = {name: p.detach().clone() for name, p in trainable.items()}
    optimizer = torch.optim.AdamW(trainable.values(), lr=1e-3, weight_decay=0.0)
    actor.model.zero_grad(set_to_none=True)
    scores = actor.get_token_log_probs(
        decision["prompt_ids"], decision["token_ids"], decision["token_choices"],
    )
    loss = -scores.sum()
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in trainable.values() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g).item() for g in gradients)
    assert all(p.grad is None for p, _ in frozen)
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    assert any(not torch.equal(p.detach(), initial[name]) for name, p in trainable.items())
    assert all(p._version == version for p, version in frozen)
    assert all(torch.isfinite(p).all() for p in trainable.values())
    actor.model.zero_grad(set_to_none=True)
    with torch.no_grad():
        updated = actor.get_token_log_probs(
            decision["prompt_ids"], decision["token_ids"], decision["token_choices"],
        ).sum().item()
    assert math.isfinite(updated)
    return {
        "trainable_parameters": sum(p.numel() for p in trainable.values()),
        "total_parameters": sum(p.numel() for p in actor.model.parameters()),
        "initialization_and_rollout_probability_checks_passed": True,
        "base_gradients_absent": True, "base_parameter_versions_unchanged": True,
        "adapters_changed": True, "gradient_norm_before_clipping": grad_norm.item(),
        "log_prob_before": decision["log_prob"], "log_prob_after": updated,
    }


def test_lora_initialization_update_and_checkpoint(policy, observation, monkeypatch, tmp_path):
    actor = WikiGamePolicy(model=deepcopy(policy.model), tokenizer=policy.tokenizer)
    original_weights = [(p, p.detach().clone()) for p in actor.model.parameters()]
    force_action(actor, observation, 10, monkeypatch)
    decision = actor.choose_action(observation)
    report = check_lora_update(actor, observation, decision, atol=1e-5)
    assert report["log_prob_after"] > report["log_prob_before"]
    assert all(torch.equal(p.detach(), original) for p, original in original_weights)
    with pytest.raises(ValueError, match="already has adapters"):
        actor.enable_lora()

    checkpoint = tmp_path / "adapter"
    actor.model.save_pretrained(checkpoint, save_embedding_layers=False)
    assert (checkpoint / "adapter_config.json").is_file()
    assert (checkpoint / "adapter_model.safetensors").is_file()
    assert not (checkpoint / "model.safetensors").exists()
    base = actor.model.unload()
    assert all(torch.equal(p.detach(), original) for p, original in original_weights)
    reloaded = WikiGamePolicy(model=base, tokenizer=policy.tokenizer)
    reloaded.model = PeftModel.from_pretrained(reloaded.model, checkpoint, is_trainable=True)
    reloaded.model.eval()
    assert any(p.requires_grad for p in reloaded.model.parameters())
    with torch.no_grad():
        rescored = reloaded.get_token_log_probs(
            decision["prompt_ids"], decision["token_ids"], decision["token_choices"],
        ).sum().item()
    assert rescored == pytest.approx(report["log_prob_after"], abs=1e-5)
    force_action(reloaded, observation, 10, monkeypatch)
    assert_decision(reloaded, observation, reloaded.choose_action(observation))


@pytest.mark.integration
def test_pretrained_lora_update(real_policy, observation, monkeypatch, tmp_path):
    """Exercise the actual actor on its selected device without changing saved weights."""
    actor = real_policy
    original_flags = [(p, p.requires_grad) for p in actor.model.parameters()]
    force_action(actor, observation, 10, monkeypatch)
    decision = actor.choose_action(observation)
    try:
        if actor.model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(actor.model.device)
        report = check_lora_update(actor, observation, decision, atol=1e-4)
        checkpoint = tmp_path / "adapter"
        actor.model.save_pretrained(checkpoint, save_embedding_layers=False)
        actor.model = PeftModel.from_pretrained(actor.model.unload(), checkpoint, is_trainable=True)
        actor.model.eval()
        with torch.no_grad():
            reloaded = actor.get_token_log_probs(
                decision["prompt_ids"], decision["token_ids"], decision["token_choices"],
            ).sum().item()
        assert reloaded == pytest.approx(report["log_prob_after"], abs=1e-4)
        force_action(actor, observation, 10, monkeypatch)
        assert_decision(actor, observation, actor.choose_action(observation), atol=1e-4)
        report["adapter_reload_probability_check_passed"] = True
        report["device"] = str(actor.model.device)
        report["base_dtype"] = str(next(actor.model.get_base_model().parameters()).dtype)
        if actor.model.device.type == "cuda":
            report["peak_allocated_gib"] = torch.cuda.max_memory_allocated(actor.model.device) / 2**30
        print("LoRA training check:", json.dumps(report, indent=2))
    finally:
        if isinstance(actor.model, PeftModel):
            actor.model = actor.model.unload()
        for parameter, flag in original_flags:
            parameter.requires_grad_(flag)
        actor.model.eval()


def controlled_grpo_group(actor, observation, monkeypatch):
    """Synthetic outcomes isolate the optimizer math from navigation quality."""
    group = []
    for action, reward in [(10, 1.0), (300, 0.0)]:
        force_action(actor, observation, action, monkeypatch)
        group.append({
            "start": "Europe", "target": "France", "episode_return": reward,
            "steps": [actor.choose_action(observation)],
        })
    return group


def test_streamed_grpo_matches_full_graph_update(policy, observation, monkeypatch):
    actor = WikiGamePolicy(model=deepcopy(policy.model), tokenizer=policy.tokenizer)
    actor.enable_lora()
    actor.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    group = controlled_grpo_group(actor, observation, monkeypatch)
    constant = deepcopy(group)
    for episode in constant:
        episode["episode_return"] = 0.0
    groups = [group, constant]
    reference = WikiGamePolicy(model=deepcopy(actor.model), tokenizer=policy.tokenizer)
    reference.model.train()
    loss = 0
    advantages = group_advantages([1.0, 0.0])
    for episode, advantage in zip(group, advantages):
        step = episode["steps"][0]
        current = reference.get_token_log_probs(step["prompt_ids"], step["token_ids"], step["token_choices"])
        mask = torch.tensor([len(choices) > 1 for choices in step["token_choices"]])
        old = torch.tensor(step["token_log_probs"])
        # Two active routes and two zero-advantage routes: denominator remains four.
        loss = loss + clipped_token_loss(current[mask], old[mask], advantage).mean() / 4
    loss.backward()
    expected = {name: p.grad.detach().clone() for name, p in reference.model.named_parameters() if p.grad is not None}
    optimizer = torch.optim.AdamW((p for p in actor.model.parameters() if p.requires_grad), lr=1e-3, weight_decay=0)
    step_optimizer = optimizer.step
    captured = {}

    def record_gradients():
        captured.update({name: p.grad.detach().clone() for name, p in actor.model.named_parameters() if p.grad is not None})
        step_optimizer()

    monkeypatch.setattr(optimizer, "step", record_gradients)
    frozen = [(p, p.detach().clone()) for p in actor.model.parameters() if not p.requires_grad]
    result = grpo_update(actor, groups, optimizer, max_grad_norm=1e9, atol=1e-5)
    assert captured.keys() == expected.keys()
    for name in expected:
        assert torch.allclose(captured[name], expected[name], atol=1e-6, rtol=1e-5), name
    assert result["loss"] == pytest.approx(loss.item(), abs=1e-6)
    assert result["optimizer_step_applied"] and result["adapters_changed"]
    assert result["initial_probability_checks_passed"]
    assert result["gradient_norm_before_clipping"] > 0
    assert result["zero_variance_groups"] == 1 and result["routes"] == 4
    assert all(torch.equal(p.detach(), original) for p, original in frozen)
    assert not actor.model.training
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_grpo_skips_constant_reward_groups(policy, observation, monkeypatch):
    actor = WikiGamePolicy(model=deepcopy(policy.model), tokenizer=policy.tokenizer)
    actor.enable_lora()
    group = controlled_grpo_group(actor, observation, monkeypatch)
    for episode in group:
        episode["episode_return"] = 1.09
    optimizer = torch.optim.AdamW((p for p in actor.model.parameters() if p.requires_grad), lr=1e-5)

    def unexpected(*args, **kwargs):
        raise AssertionError("Constant returns should not trigger scoring or an optimizer step.")

    monkeypatch.setattr(actor, "_next_log_probs", unexpected)
    monkeypatch.setattr(optimizer, "step", unexpected)
    result = grpo_update(actor, [group], optimizer)
    assert not result["optimizer_step_applied"]
    assert result["initial_probability_checks_passed"] is None
    assert result["zero_variance_groups"] == 1 and not optimizer.state


def test_grpo_rejects_stale_probabilities_before_stepping(policy, observation, monkeypatch):
    actor = WikiGamePolicy(model=deepcopy(policy.model), tokenizer=policy.tokenizer)
    actor.enable_lora()
    group = controlled_grpo_group(actor, observation, monkeypatch)
    step = group[-1]["steps"][0]
    position = next(i for i, choices in enumerate(step["token_choices"]) if len(choices) > 1)
    step["token_log_probs"][position] -= 0.5
    before = [(p, p.detach().clone()) for p in actor.model.parameters()]
    optimizer = torch.optim.AdamW((p for p in actor.model.parameters() if p.requires_grad), lr=1e-5)
    with pytest.raises(RuntimeError, match="Behavior log probabilities do not match"):
        grpo_update(actor, [group], optimizer)
    assert not optimizer.state
    assert all(torch.equal(p.detach(), original) for p, original in before)
    assert all(p.grad is None for p in actor.model.parameters())
    assert not actor.model.training


def test_grpo_cli_collects_audits_and_reports_update(policy, tmp_path, monkeypatch):
    from training import grpo

    actor = WikiGamePolicy(model=deepcopy(policy.model), tokenizer=policy.tokenizer)
    articles = {
        "Start": {"title": "Start", "text": "Start", "outgoing": ["Mid", "Trap"]},
        "Mid": {"title": "Mid", "text": "Mid", "outgoing": ["Goal"]},
        "Trap": {"title": "Trap", "text": "Trap", "outgoing": ["Trap"]},
        "Goal": {"title": "Goal", "text": "Goal", "outgoing": ["Start"]},
    }
    article_path, tasks_path = tmp_path / "articles.json", tmp_path / "train.json"
    article_path.write_text(json.dumps(articles), encoding="utf-8")
    tasks_path.write_text(json.dumps([
        {"start": "Start", "target": "Goal", "shortest_distance": 2},
        {"start": "Mid", "target": "Goal", "shortest_distance": 1},
    ]), encoding="utf-8")
    output = tmp_path / "run"
    monkeypatch.setattr(grpo, "WikiGamePolicy", lambda **kwargs: actor)
    monkeypatch.setattr(sys, "argv", [
        "grpo", "--tasks", str(tasks_path), "--articles", str(article_path),
        "--distance", "2", "--limit", "1", "--rollouts", "4", "--max-clicks", "2", "--output", str(output),
    ])
    grpo.main()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["route_audit"]["all_recorded_transitions_and_returns_valid"]
    assert report["group_summary"]["total_rollouts"] == 4
    assert report["settings"]["algorithm"] == "constrained_token_grpo"
    assert report["settings"]["kl_beta"] == 0
    assert all(e["start"] == "Start" and "steps" not in e for e in report["episodes"])
    assert report["update"]["optimizer_step_applied"] == bool(report["group_summary"]["groups_with_reward_variation"])
    if report["update"]["optimizer_step_applied"]:
        assert (output / "adapter" / "adapter_model.safetensors").exists()
    else:
        assert report["adapter_checkpoint"] is None
    with pytest.raises(SystemExit):
        grpo.main()  # Existing run directories cannot be overwritten.


@pytest.mark.integration
def test_pretrained_grpo_checkpointed_long_context(real_policy, observation, monkeypatch):
    actor = real_policy
    original_flags = [(p, p.requires_grad) for p in actor.model.parameters()]
    observation["text"] = "Article context. " * 2000
    try:
        actor.enable_lora()
        actor.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        group = controlled_grpo_group(actor, observation, monkeypatch)
        assert all(len(e["steps"][0]["prompt_ids"]) > 3000 for e in group)
        optimizer = torch.optim.AdamW((p for p in actor.model.parameters() if p.requires_grad), lr=1e-5, weight_decay=0)
        if actor.model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(actor.model.device)
        report = grpo_update(actor, [group], optimizer)
        assert report["optimizer_step_applied"] and report["adapters_changed"]
        assert report["initial_probability_checks_passed"]
        assert report["base_parameter_versions_unchanged"]
        if actor.model.device.type == "cuda":
            report["peak_allocated_gib"] = torch.cuda.max_memory_allocated(actor.model.device) / 2**30
        print("Checkpointed GRPO check:", json.dumps(report, indent=2))
    finally:
        if isinstance(actor.model, PeftModel):
            actor.model = actor.model.unload()
        actor.model.gradient_checkpointing_disable()
        for parameter, flag in original_flags:
            parameter.requires_grad_(flag)
        actor.model.eval()
