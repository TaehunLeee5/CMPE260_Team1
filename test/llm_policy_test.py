from copy import deepcopy
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
    tokens = iter(policy._action_sequences(policy._tool_calls(observation))[action])

    def draw(probabilities, num_samples=1, **kwargs):
        token = next(tokens)
        assert probabilities[0, token] > 0
        return torch.tensor([[token]], device=probabilities.device)

    monkeypatch.setattr(torch, "multinomial", draw)
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
    decision = policy.choose_action(observation)
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
    print("\\nTool:", decision["tool_call"], "log probability:", decision["log_prob"])


@pytest.mark.integration
@pytest.mark.parametrize("action", [10, 100, 300])
def test_pretrained_controlled_calls(real_policy, observation, monkeypatch, action):
    tokens = iter(real_policy._action_sequences(real_policy._tool_calls(observation))[action])

    def draw(probabilities, num_samples=1, **kwargs):
        token = next(tokens)
        assert probabilities[0, token] > 0
        return torch.tensor([[token]], device=probabilities.device)

    monkeypatch.setattr(torch, "multinomial", draw)
    decision = real_policy.choose_action(observation)
    assert decision["action"] == action
    assert_decision(real_policy, observation, decision, atol=1e-4)
