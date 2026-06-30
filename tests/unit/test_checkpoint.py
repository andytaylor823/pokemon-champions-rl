"""Unit tests for checkpoint save/load (the neutral persistence seam).

Covers: round-trip weight fidelity, config rebuild from the stored dict, the
lean (no-optimizer) vs rich (resume) payloads, and the fail-loud guards.
"""

from __future__ import annotations

import pytest
import torch

from checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointFormatError,
    load_checkpoint,
    save_checkpoint,
)
from cvpn import CVPN, CVPNConfig

# A tiny CVPN keeps these tests fast.
_SMALL = CVPNConfig(d_model=32, n_heads=2, n_layers=1, ffn_mult=2)


def _state_dicts_equal(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def test_save_load_round_trip_preserves_weights_and_generation(tmp_path):
    net = CVPN(_SMALL)
    optim = torch.optim.AdamW(net.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, net=net, generation=7, optimizer=optim, trainer_config={"learning_rate": 1e-3}, rng_state={"x": 1})
    loaded = load_checkpoint(path)

    assert loaded.generation == 7
    assert _state_dicts_equal(net.state_dict(), loaded.net.state_dict())
    assert loaded.optimizer_state_dict is not None
    assert loaded.trainer_config == {"learning_rate": 1e-3}
    assert loaded.rng_state == {"x": 1}


def test_load_rebuilds_net_from_stored_config(tmp_path):
    """A net saved with a non-default architecture reloads with that architecture."""
    net = CVPN(_SMALL)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, net=net, generation=0)

    loaded = load_checkpoint(path)
    assert loaded.net.config.d_model == 32
    assert loaded.net.config.n_layers == 1
    # Rebuilt net must accept the saved weights (same shapes).
    assert _state_dicts_equal(net.state_dict(), loaded.net.state_dict())


def test_lean_checkpoint_has_no_optimizer(tmp_path):
    """A published (weights-only) checkpoint loads with None resume fields."""
    net = CVPN(_SMALL)
    path = tmp_path / "lean.pt"
    save_checkpoint(path, net=net, generation=3)

    loaded = load_checkpoint(path)
    assert loaded.generation == 3
    assert loaded.optimizer_state_dict is None
    assert loaded.trainer_config is None
    assert loaded.rng_state is None


def test_format_version_mismatch_raises(tmp_path):
    net = CVPN(_SMALL)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, net=net, generation=1)

    # Tamper the format version on disk.
    payload = torch.load(path, weights_only=False)
    payload["format_version"] = CHECKPOINT_FORMAT_VERSION + 99
    torch.save(payload, path)

    with pytest.raises(CheckpointFormatError, match="format_version"):
        load_checkpoint(path)


def test_missing_required_key_raises(tmp_path):
    path = tmp_path / "corrupt.pt"
    torch.save({"format_version": CHECKPOINT_FORMAT_VERSION, "generation": 0}, path)  # no model_state_dict/config

    with pytest.raises(CheckpointFormatError, match="missing key"):
        load_checkpoint(path)
