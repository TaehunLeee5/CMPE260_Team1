from copy import deepcopy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from transformers import Qwen3Config, Qwen3ForCausalLM, set_seed

from training.checkpoint import (capture_rng, read_checkpoint, resolve_resume, restore_rng,
                                 restore_training_state, save_checkpoint)
from training.grpo import validate


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_checkpoint_restores_next_stochastic_step_and_optimizer(tmp_path, device):
    previous_rng = capture_rng()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        set_seed(42)
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                            head_dim=8, max_position_embeddings=64, attention_dropout=0.0)
        base = Qwen3ForCausalLM(config).to(device)
        original_base = deepcopy(base.state_dict())
        model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4,
                                               target_modules="all-linear", lora_dropout=0.0))
        model.eval()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)

        def step(actor, optim):
            size = random.randint(4, 8)
            shift = int(np.random.randint(0, 32))
            inputs = (torch.randint(0, 32, (1, size), device=device) + shift) % 32
            optim.zero_grad(set_to_none=True)
            loss = actor(input_ids=inputs, labels=inputs, use_cache=False).loss
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            return loss.item()

        step(model, optimizer)
        run = tmp_path / "run"
        checkpoint = run / "checkpoints" / "batch_000001"
        save_checkpoint(SimpleNamespace(model=model), optimizer, checkpoint, {"batch": 1})
        expected_loss = step(model, optimizer)
        expected_weights = {name: value.detach().clone() for name, value in get_peft_model_state_dict(model).items()}
        expected_optimizer = deepcopy(optimizer.state_dict())

        restored_base = Qwen3ForCausalLM(config).to(device)
        restored_base.load_state_dict(original_base)
        restored = PeftModel.from_pretrained(restored_base, checkpoint / "adapter", is_trainable=True)
        restored.eval()
        restored_optimizer = torch.optim.AdamW((p for p in restored.parameters() if p.requires_grad), lr=0.99)
        state = read_checkpoint(checkpoint)
        restore_training_state(SimpleNamespace(model=restored), restored_optimizer, state)
        assert step(restored, restored_optimizer) == pytest.approx(expected_loss, abs=1e-7)
        actual = get_peft_model_state_dict(restored)
        assert all(torch.equal(actual[name], value) for name, value in expected_weights.items())
        assert restored_optimizer.param_groups[0]["lr"] == 1e-3
        actual_optimizer = restored_optimizer.state_dict()
        for key, values in expected_optimizer["state"].items():
            assert all(torch.equal(actual_optimizer["state"][key][name], value) for name, value in values.items())
        assert resolve_resume(run) == (run, checkpoint)
        assert resolve_resume(checkpoint) == (run, checkpoint)
        (run / "checkpoints" / "batch_000002").mkdir()  # An incomplete folder is not resumable.
        assert resolve_resume(run) == (run, checkpoint)
        with pytest.raises(FileExistsError):
            save_checkpoint(SimpleNamespace(model=model), optimizer, checkpoint, {})
    finally:
        restore_rng(previous_rng)
        torch.set_num_threads(threads)


def test_validation_preserves_all_training_random_streams(monkeypatch):
    from training import grpo

    previous = capture_rng()
    try:
        set_seed(42)
        saved = capture_rng()
        expected = (random.random(), float(np.random.random()), torch.rand(4))
        restore_rng(saved)

        def consuming_validation(*args, **kwargs):
            random.random()
            np.random.random(10)
            torch.rand(12)
            return {"summary": "fixed validation"}

        monkeypatch.setattr(grpo, "evaluate", consuming_validation)
        assert validate(None, []) == {"summary": "fixed validation"}
        actual = (random.random(), float(np.random.random()), torch.rand(4))
        assert actual[:2] == expected[:2]
        assert torch.equal(actual[2], expected[2])
    finally:
        restore_rng(previous)
