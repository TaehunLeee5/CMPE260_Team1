import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def capture_rng():
    numpy = np.random.get_state()
    return {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": {"name": numpy[0], "keys": torch.from_numpy(numpy[1].astype(np.int64)),
                  "position": numpy[2], "has_gauss": numpy[3], "cached_gauss": numpy[4]},
    }


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    numpy = state["numpy"]
    np.random.set_state((numpy["name"], numpy["keys"].numpy().astype(np.uint32),
                         numpy["position"], numpy["has_gauss"], numpy["cached_gauss"]))
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same CUDA device count as the saved run.")
        torch.cuda.set_rng_state_all(state["cuda"])


def write_json(path, value):
    """Replace a small progress file atomically on Windows and Linux."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(policy, optimizer, destination, state, report=None):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Checkpoint already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".writing_", dir=destination.parent) as temporary:
        staging = Path(temporary) / "ready"
        staging.mkdir()
        policy.model.save_pretrained(staging / "adapter", save_embedding_layers=False)
        payload = {
            "format_version": 1, "progress": state, "optimizer": optimizer.state_dict(),
            "parameter_names": [name for name, p in policy.model.named_parameters() if p.requires_grad],
            "rng": capture_rng(),
        }
        torch.save(payload, staging / "trainer_state.pt")
        if report is not None:
            write_json(staging / "report.json", report)
        staging.rename(destination)


def read_checkpoint(path):
    state = torch.load(Path(path) / "trainer_state.pt", map_location="cpu", weights_only=True)
    if state["format_version"] != 1:
        raise ValueError("Unsupported training checkpoint format.")
    return state


def restore_training_state(policy, optimizer, state):
    names = [name for name, p in policy.model.named_parameters() if p.requires_grad]
    if names != state["parameter_names"]:
        raise ValueError("Trainable parameter names/order do not match the checkpoint.")
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])


def resolve_resume(path):
    """Accept a run directory or its latest complete checkpoint directory."""
    path = Path(path)
    run = path.parent.parent if (path / "trainer_state.pt").is_file() else path
    checkpoints = sorted(p for p in (run / "checkpoints").glob("batch_*")
                         if (p / "trainer_state.pt").is_file())
    if not checkpoints:
        raise ValueError(f"No completed batch checkpoint found under {run}")
    latest = checkpoints[-1]
    if path != run and path.resolve() != latest.resolve():
        raise ValueError("Resume the latest checkpoint to preserve this run's batch history.")
    return run, latest
