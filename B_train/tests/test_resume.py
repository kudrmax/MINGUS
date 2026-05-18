"""Test checkpoint save/load round-trip and resume logic."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from B_train._resume import (
    CheckpointState,
    load_checkpoint,
    save_checkpoint,
    is_phase_done,
)


def _make_dummy_model():
    return nn.Linear(4, 2)


def test_save_and_load_round_trip(tmp_path):
    model = _make_dummy_model()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    state = CheckpointState(epoch=3, best_val_loss=1.234)

    save_checkpoint(tmp_path, "pitch", state, model, optim)

    model2 = _make_dummy_model()
    optim2 = torch.optim.SGD(model2.parameters(), lr=0.1)
    loaded = load_checkpoint(tmp_path, "pitch", model2, optim2)

    assert loaded.epoch == 3
    assert loaded.best_val_loss == pytest.approx(1.234)
    # Weights must match
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)


def test_load_returns_none_when_missing(tmp_path):
    model = _make_dummy_model()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    assert load_checkpoint(tmp_path, "pitch", model, optim) is None


def test_is_phase_done_flag(tmp_path):
    assert not is_phase_done(tmp_path, "pitch")
    (tmp_path / "pitch.done").touch()
    assert is_phase_done(tmp_path, "pitch")
