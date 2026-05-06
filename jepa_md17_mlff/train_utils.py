from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def update_ema(online: torch.nn.Module, target: torch.nn.Module, decay: float) -> None:
    for p_online, p_target in zip(online.parameters(), target.parameters()):
        p_target.data.mul_(decay).add_(p_online.data, alpha=1.0 - decay)


def random_atom_mask(batch_size: int, n_atoms: int, ratio: float, device: torch.device) -> torch.Tensor:
    mask = torch.rand(batch_size, n_atoms, device=device) < ratio
    empty = ~mask.any(dim=1)
    if empty.any():
        chosen = torch.randint(0, n_atoms, (int(empty.sum()),), device=device)
        mask[empty, chosen] = True
    return mask


class MetricAverager:
    def __init__(self):
        self.totals: dict[str, float] = {}
        self.count = 0

    def update(self, metrics: dict[str, float], n: int = 1) -> None:
        for key, value in metrics.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value) * n
        self.count += n

    def compute(self) -> dict[str, float]:
        denom = max(self.count, 1)
        return {key: value / denom for key, value in self.totals.items()}
