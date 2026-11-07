"""Policy mechanics use a tiny random actor; integration tests load real Qwen."""

from copy import deepcopy
from collections import Counter
import json
import math
import string

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from environment.environment import WikiGameEnv
from environment.episode_runner import EpisodeRunner
from policy.llm_policy import WikiGamePolicy


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
