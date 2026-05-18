"""Checkpoint save/load + per-phase done flag for MINGUS train.py.

MINGUS trains two phases sequentially (pitch model, then duration model).
Colab can drop the session at any time — we want to resume:
  * mid-phase: continue from last completed epoch in that phase
  * past phase 1: skip pitch entirely, start duration

Checkpoint files (per phase) inside <work_dir>:
  <phase>_state.pt       — model state_dict + optimizer state_dict + epoch + best_val_loss
  <phase>.done           — empty marker file written when phase finishes successfully
  <phase>_best.pt        — best-val state_dict (final artefact)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class CheckpointState:
    epoch: int
    best_val_loss: float


def save_checkpoint(
    work_dir: Path,
    phase: str,
    state: CheckpointState,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": state.epoch,
            "best_val_loss": state.best_val_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        work_dir / f"{phase}_state.pt",
    )


def load_checkpoint(
    work_dir: Path,
    phase: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Optional[CheckpointState]:
    work_dir = Path(work_dir)
    path = work_dir / f"{phase}_state.pt"
    if not path.exists():
        return None
    blob = torch.load(path, map_location="cpu")
    model.load_state_dict(blob["model"])
    optimizer.load_state_dict(blob["optimizer"])
    return CheckpointState(epoch=blob["epoch"], best_val_loss=blob["best_val_loss"])


def mark_phase_done(work_dir: Path, phase: str) -> None:
    Path(work_dir, f"{phase}.done").touch()


def is_phase_done(work_dir: Path, phase: str) -> bool:
    return Path(work_dir, f"{phase}.done").exists()


def save_best(
    work_dir: Path, phase: str, model: torch.nn.Module
) -> None:
    torch.save(model.state_dict(), Path(work_dir, f"{phase}_best.pt"))
