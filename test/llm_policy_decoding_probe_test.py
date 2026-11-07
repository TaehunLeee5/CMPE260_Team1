"""Paired decoding diagnostics; PASS means collection succeeded, not target success."""
import copy
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re

import pytest
import torch
import transformers

from environment.environment import WikiGameEnv
from policy.llm_policy import WikiGamePolicy

ROOT = Path(__file__).resolve().parents[1]
TOKEN_CAP = int(os.environ.get('WIKI_PROBE_INPUT_TOKENS', '4096'))
FREE_TOKENS = int(os.environ.get('WIKI_PROBE_FREE_TOKENS', '16'))
if TOKEN_CAP < 1 or FREE_TOKENS < 1:
    raise pytest.UsageError('WIKI_PROBE_* token budgets must be positive.')


def leading_number(text):
    """Diagnostic extraction only; never used to select an environment action."""
    match = re.match(r'\s*(\d+)', text)
    return int(match.group(1)) if match else None


def continuations(sequences, prefix):
    return sorted({s[len(prefix)] for s in sequences.values()
                   if len(s) > len(prefix) and s[:len(prefix)] == prefix})


@pytest.fixture(scope='session')
def probe_report():
    report = {'python': platform.python_version(), 'torch': torch.__version__,
              'transformers': transformers.__version__, 'input_token_cap': TOKEN_CAP,
              'free_output_budget': FREE_TOKENS, 'cases': [],
              'note': 'Successful diagnostic tests do not imply correct ID selection.'}
    yield report
    destination = ROOT / 'evaluation/results/policy_decoding_probe.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'\nDecoding probe report: {destination}')


@pytest.fixture(scope='session')
def probe_policy(probe_report):
    policy = WikiGamePolicy()
    colon = policy.tokenizer.encode(':', add_special_tokens=False)
    assert len(colon) == 1 and getattr(policy, 'action_end_token_id', None) == colon[0], (
        'Apply the colon protocol changes before this diagnostic.'
    )
    probe_report['action_end_token_id'] = policy.action_end_token_id
    policy.max_input_tokens = min(policy.max_input_tokens or TOKEN_CAP, TOKEN_CAP)
    policy.model.eval()
    probe_report.update(model=policy.model.name_or_path, dtype=str(policy.model.dtype),
                        device=str(policy.model.device), input_tokens=policy.max_input_tokens)
    return policy


@pytest.fixture(scope='session')
def largest_observation():
    path = ROOT / 'data/processed/articles.json'
    if not path.exists():
        pytest.fail(f'Real articles required: {path}')
    env = WikiGameEnv(path, max_clicks=20)
    try:
        current = max(env.articles, key=lambda a: len(env.articles[a]['outgoing']))
        target = next((a for a in env.articles[current]['outgoing'] if a != current), None)
        assert target is not None
        env.reset(options={'start': current, 'target': target})
        return copy.deepcopy(env.get_llm_observation())
    finally:
        env.close()


def raw_prefix_probe(policy, inputs, sequences, prefix, chosen=None):
    device = inputs['input_ids'].device
    suffix = torch.tensor([prefix], dtype=inputs['input_ids'].dtype, device=device)
    ids = torch.cat([inputs['input_ids'], suffix], dim=1)
    mask = inputs.get('attention_mask', torch.ones_like(inputs['input_ids']))
    mask = torch.cat([mask, torch.ones_like(suffix)], dim=1)
    with torch.no_grad():
        logits = policy.model(input_ids=ids, attention_mask=mask,
                              use_cache=False, logits_to_keep=1).logits[0, -1].float()
        probabilities = torch.softmax(logits, dim=-1)
    allowed = continuations(sequences, prefix)
    assert allowed, f'No continuation for a constrained prefix: {prefix}'
    conditional = torch.softmax(logits[allowed], dim=-1)
    top_values, top_ids = probabilities.topk(12)

    def token_info(token, probability):
        return {'id': token,
                'text': policy.tokenizer.decode([token], skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False),
                'raw_probability': float(probability)}

    entries = []
    for i, token in enumerate(allowed):
        entry = token_info(token, probabilities[token].item())
        entry['conditional_probability_after_mask'] = conditional[i].item()
        entries.append(entry)
    selected = allowed[conditional.argmax().item()]
    top_token = top_ids[0].item()
    result = {'prefix_tokens': prefix,
              'prefix_text': policy.tokenizer.decode(prefix, skip_special_tokens=False),
              'chosen_next_token': chosen, 'predicted_next_under_mask': selected,
              'raw_top_token_allowed': top_token in allowed,
              'allowed_raw_probability_mass': probabilities[allowed].sum().item(),
              'eos_raw_probability': probabilities[policy.tokenizer.eos_token_id].item(),
              'allowed_tokens': entries,
              'raw_top_tokens': [token_info(t, p) for t, p in zip(top_ids.tolist(), top_values.tolist())]}
    # Record differences; independently replayed forward kernels may differ by
    # precision. A difference is evidence to inspect, not silently discard.
    if chosen is not None:
        result['chosen_matches_replayed_mask_argmax'] = chosen == selected
    return result


@pytest.mark.parametrize('target_id', [0, 9, 10, 99, 100, 293])
def test_paired_decoding_diagnostic(probe_policy, probe_report, largest_observation,
                                   monkeypatch, target_id):
    p = probe_policy
    obs = copy.deepcopy(largest_observation)
    titles = {a['id']: a['title'] for a in obs['actions']}
    if target_id not in titles:
        pytest.skip(f'Largest article has no ID {target_id}.')
    obs['target_title'] = titles[target_id]
    expected = {a['id']: p.tokenizer.encode(str(a['id']), add_special_tokens=False)
                + [p.action_end_token_id] for a in obs['actions']}
    captured = {}
    original_generate = p.model.generate

    @functools.wraps(original_generate)
    def capture(*args, **kwargs):
        captured['inputs'] = {k: v.detach().clone() for k, v in kwargs.items()
                              if torch.is_tensor(v)}
        captured['configuration'] = copy.deepcopy(kwargs['generation_config'])
        result = original_generate(*args, **kwargs)
        captured['result'] = result
        return result

    with monkeypatch.context() as patch:
        patch.setattr(p.model, 'generate', capture)
        constrained_id, rollout_lp = p.choose_action(obs, deterministic=True)
    inputs = captured['inputs']
    length = inputs['input_ids'].shape[1]
    constrained_tokens = captured['result'].sequences[0, length:].tolist()
    assert constrained_id in expected
    assert constrained_tokens == expected[constrained_id]
    with torch.no_grad():
        rescored_lp = p.get_action_log_prob(obs, constrained_id).item()
    assert math.isfinite(rollout_lp) and math.isfinite(rescored_lp)
    assert abs(rollout_lp - rescored_lp) <= 1e-4

    # Reuse the exact input tensors captured above, not a rebuilt prompt.
    # Preserve the generation configuration except for extra diagnostic output
    # room, and restore normal chat EOS for unrestricted diagnostic output.
    free_config = copy.deepcopy(captured['configuration'])
    # Unrestricted diagnostics use the normal chat EOS so any title/extra text
    # after the colon remains visible. The constrained policy stops at colon.
    free_config.eos_token_id = p.tokenizer.eos_token_id
    headroom = p.model.config.max_position_embeddings - length
    assert headroom > 0
    free_config.max_new_tokens = min(FREE_TOKENS, headroom)
    assert free_config.do_sample is False
    with torch.no_grad():
        unrestricted = original_generate(**inputs, generation_config=free_config,
                                         logits_to_keep=1)
    unrestricted_tokens = unrestricted.sequences[0, length:].tolist()
    assert torch.equal(unrestricted.sequences[0, :length], inputs['input_ids'][0])
    free_text = p.tokenizer.decode(unrestricted_tokens, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
    first_difference = next((i for i, (a, b) in enumerate(zip(constrained_tokens, unrestricted_tokens))
                             if a != b), None)
    if first_difference is None and len(constrained_tokens) != len(unrestricted_tokens):
        first_difference = min(len(constrained_tokens), len(unrestricted_tokens))
    trace = [raw_prefix_probe(p, inputs, expected, constrained_tokens[:i], token)
             for i, token in enumerate(constrained_tokens)]
    correct_digits = p.tokenizer.encode(str(target_id), add_special_tokens=False)
    stop_probe = raw_prefix_probe(p, inputs, expected, correct_digits)
    prompt_hash = hashlib.sha256(
        json.dumps(inputs['input_ids'][0].tolist()).encode('utf-8')
    ).hexdigest()
    record = {
        'current_article': obs['current_title'], 'link_count': len(titles),
        'expected_id': target_id, 'expected_title': titles[target_id],
        'prompt_tokens': length, 'shared_prompt_sha256': prompt_hash,
        'constrained_id': constrained_id, 'constrained_title': titles[constrained_id],
        'constrained_correct': constrained_id == target_id,
        'constrained_generated_tokens': constrained_tokens,
        'constrained_log_probability': rollout_lp,
        'constrained_log_probability_error': abs(rollout_lp - rescored_lp),
        'unrestricted_text': free_text,
        'unrestricted_text_with_specials': p.tokenizer.decode(unrestricted_tokens, skip_special_tokens=False,
                                                             clean_up_tokenization_spaces=False),
        'unrestricted_tokens': unrestricted_tokens,
        'unrestricted_leading_number': leading_number(free_text),
        'unrestricted_leading_number_correct': leading_number(free_text) == target_id,
        'unrestricted_hit_token_limit': len(unrestricted_tokens) >= free_config.max_new_tokens
                                       and (not unrestricted_tokens or unrestricted_tokens[-1] != p.tokenizer.eos_token_id),
        'first_different_output_token_index': first_difference,
        'constrained_prefix_raw_logits': trace,
        'raw_logits_after_teacher_forced_correct_id': stop_probe,
        'requested_constrained_config': captured['configuration'].to_dict(),
        'requested_unrestricted_config': free_config.to_dict(),
    }
    probe_report['cases'].append(record)
    # Save after each completed case too, so completed evidence survives interruption.
    destination = ROOT / 'evaluation/results/policy_decoding_probe.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(probe_report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    excluded = [t for t in trace if not t['raw_top_token_allowed']]
    print(f'\nExpected {target_id}: {titles[target_id]} | constrained={constrained_id} '
          f'| unrestricted={free_text!r} | steps excluding raw top token={len(excluded)}')
