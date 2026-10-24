import math
from copy import deepcopy

import pytest
import torch

from policy.llm_policy import WikiGamePolicy
from policy.prompt import SYSTEM_PROMPT


@pytest.fixture(scope="session")
def wiki_policy():
    """Load the actual Qwen model once through WikiGamePolicy."""
    return WikiGamePolicy()


@pytest.fixture
def observation():
    """Provide the same readable fields returned by the environment."""
    return {
        "current_title": "Europe",
        "target_title": "France",
        "text": "Europe is a continent that includes many countries.",
        "history": ["Europe"],
        "remaining_clicks": 10,
        "actions": [
            {"id": 1, "title": "Germany"},
            {"id": 10, "title": "Italy"},
            {"id": 100, "title": "France"},
        ],
    }


def test_build_prompt(wiki_policy, observation):
    """Fill the template without altering article text or observation data."""
    observation["history"] = ["Earth", "Europe"]
    observation["text"] = "Article content containing {target_title} and {links}."
    original = deepcopy(observation)

    prompt = wiki_policy.build_prompt(observation)

    assert "Target article: France" in prompt
    assert "Current article: Europe" in prompt
    assert "Remaining clicks: 10" in prompt
    assert "Recent history: Earth -> Europe" in prompt
    assert observation["text"] in prompt
    for action in observation["actions"]:
        assert f"{action['id']}: {action['title']}" in prompt

    inputs = wiki_policy._prepare_inputs(observation, 4)
    formatted = wiki_policy.tokenizer.decode(inputs["input_ids"][0])
    assert f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>" in formatted
    assert f"<|im_start|>user\n{prompt}<|im_end|>" in formatted
    assert "<think>\n\n</think>" in formatted
    assert observation == original
    print(f"\nFormatted prompt:\n{formatted}")


def test_init(wiki_policy):
    """Keep the loaded model, tokenizer, and prompt settings on the policy."""
    assert wiki_policy.model.name_or_path == "Qwen/Qwen3-1.7B"
    assert wiki_policy.model.config.model_type == "qwen3"
    assert not wiki_policy.model.training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert wiki_policy.model.device.type == device
    tokens = wiki_policy.tokenizer.encode("100", add_special_tokens=False)
    assert wiki_policy.tokenizer.decode(tokens) == "100"
    assert wiki_policy.max_input_tokens is None


@pytest.mark.parametrize("deterministic", [False, True])
def test_choose_action(wiki_policy, observation, deterministic):
    """Return valid link IDs and matching rollout and training scores."""
    action, log_prob = wiki_policy.choose_action(observation, deterministic)
    with torch.no_grad():
        recomputed = wiki_policy.get_action_log_prob(observation, action)

    assert isinstance(action, int)
    assert action in {1, 10, 100}
    assert isinstance(log_prob, float)
    assert math.isfinite(log_prob)
    assert log_prob <= 0
    assert recomputed.item() == pytest.approx(log_prob, abs=1e-4)
    title = next(link["title"] for link in observation["actions"] if link["id"] == action)
    mode = "greedy" if deterministic else "sampled"
    print(f"\n{mode}: {title} (ID {action}), log probability: {log_prob:.6f}")


@pytest.mark.parametrize("action", [1, 10])
def test_rare_action_log_prob(wiki_policy, observation, monkeypatch, action):
    """Check rare sampled actions using real model scores and controlled draws."""
    tokens = iter(wiki_policy._action_sequences(observation["actions"])[action])

    def select_token(probabilities, num_samples=1, **kwargs):
        """Select a specified possible event without changing its probability."""
        token = next(tokens)
        assert probabilities[0, token] > 0
        return torch.tensor([[token]], device=probabilities.device)

    monkeypatch.setattr(torch, "multinomial", select_token)
    selected, log_prob = wiki_policy.choose_action(observation)
    with torch.no_grad():
        recomputed = wiki_policy.get_action_log_prob(observation, selected)

    assert selected == action
    assert math.isfinite(log_prob)
    assert recomputed.item() == pytest.approx(log_prob, abs=1e-4)


def test_action_probabilities_sum_to_one(wiki_policy, observation):
    """Check that the complete valid-action distribution is normalized."""
    with torch.no_grad():
        scores = torch.stack([
            wiki_policy.get_action_log_prob(observation, action["id"])
            for action in observation["actions"]
        ])

    assert scores.logsumexp(dim=0).item() == pytest.approx(0, abs=1e-5)


def test_selects_direct_target(wiki_policy, observation):
    """Check that greedy decoding selects the target when it is directly linked."""
    action, _ = wiki_policy.choose_action(observation, deterministic=True)

    assert action == 100, f"Expected France (ID 100), but the model chose ID {action}."


def test_multi_token_action(wiki_policy, observation):
    """Generate the whole ID when the only valid action needs several tokens."""
    observation["actions"] = [{"id": 100, "title": "France"}]

    action, log_prob = wiki_policy.choose_action(observation)

    assert action == 100
    assert log_prob == pytest.approx(0)


def test_log_prob_has_gradients(wiki_policy, observation):
    """Backpropagate through one real parameter to keep memory use small."""
    parameters = list(wiki_policy.model.parameters())
    original_flags = [parameter.requires_grad for parameter in parameters]
    wiki_policy.model.zero_grad(set_to_none=True)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        weight = wiki_policy.model.model.norm.weight
        weight.requires_grad_(True)

        log_prob = wiki_policy.get_action_log_prob(observation, 1)
        (-log_prob).backward()

        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all()
        assert weight.grad.abs().sum() > 0
    finally:
        wiki_policy.model.zero_grad(set_to_none=True)
        for parameter, original_flag in zip(parameters, original_flags):
            parameter.requires_grad_(original_flag)


def test_input_budget(wiki_policy, observation, monkeypatch):
    """Shorten article text while keeping the target and all actions visible."""
    empty_text = {**observation, "text": ""}
    limit = wiki_policy._prepare_inputs(empty_text, 4)["input_ids"].shape[1] + 20
    observation["text"] = "Article content. " * 500
    monkeypatch.setattr(wiki_policy, "max_input_tokens", limit)

    inputs = wiki_policy._prepare_inputs(observation, 4)
    prompt = wiki_policy.tokenizer.decode(inputs["input_ids"][0])

    assert inputs["input_ids"].shape[1] <= limit
    assert SYSTEM_PROMPT in prompt
    assert "Target article: France" in prompt
    for action in observation["actions"]:
        assert f"{action['id']}: {action['title']}" in prompt
    assert observation["text"] == "Article content. " * 500
    monkeypatch.setattr(wiki_policy, "max_input_tokens", 1)
    with pytest.raises(ValueError, match="links exceed"):
        wiki_policy._prepare_inputs(observation, 4)


def test_no_available_actions(wiki_policy, observation):
    """Reject terminal observations that have no available links."""
    observation["actions"] = []

    with pytest.raises(ValueError, match="no available actions"):
        wiki_policy.choose_action(observation)
