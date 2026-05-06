from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


KEY_ALIASES = {
    "z": ("z", "Z", "nuclear_charges", "atomic_numbers"),
    "R": ("R", "r", "coords", "coordinates", "positions"),
    "E": ("E", "e", "energies", "energy"),
    "F": ("F", "f", "forces"),
}


@dataclass(frozen=True)
class MD17Arrays:
    z: np.ndarray
    R: np.ndarray
    E: np.ndarray
    F: np.ndarray


def _first_key(data: np.lib.npyio.NpzFile, aliases: Iterable[str]) -> str:
    for key in aliases:
        if key in data.files:
            return key
    raise KeyError(f"None of {tuple(aliases)} found. Available keys: {data.files}")


def load_md17_npz(path: str | Path) -> MD17Arrays:
    path = Path(path)
    with np.load(path) as raw:
        keys = {name: _first_key(raw, aliases) for name, aliases in KEY_ALIASES.items()}
        z = np.asarray(raw[keys["z"]], dtype=np.int64)
        R = np.asarray(raw[keys["R"]], dtype=np.float32)
        E = np.asarray(raw[keys["E"]], dtype=np.float32).reshape(-1)
        F = np.asarray(raw[keys["F"]], dtype=np.float32)

    if R.ndim != 3 or R.shape[-1] != 3:
        raise ValueError(f"R must have shape (frames, atoms, 3), got {R.shape}")
    if F.shape != R.shape:
        raise ValueError(f"F must match R shape {R.shape}, got {F.shape}")
    if E.shape[0] != R.shape[0]:
        raise ValueError(f"E length {E.shape[0]} does not match frame count {R.shape[0]}")
    if z.ndim == 2:
        z = z[0]
    if z.ndim != 1 or z.shape[0] != R.shape[1]:
        raise ValueError(f"z must have shape (atoms,), got {z.shape} for R {R.shape}")

    return MD17Arrays(z=z, R=R, E=E, F=F)


def write_metadata(npz_path: str | Path, out_path: str | Path) -> dict:
    arrays = load_md17_npz(npz_path)
    npz_path = Path(npz_path)
    meta = {
        "source_file": npz_path.as_posix(),
        "frames": int(arrays.R.shape[0]),
        "atoms": int(arrays.R.shape[1]),
        "atomic_numbers": arrays.z.astype(int).tolist(),
        "coordinate_shape": list(arrays.R.shape),
        "energy_shape": list(arrays.E.shape),
        "force_shape": list(arrays.F.shape),
        "energy_mean": float(arrays.E.mean()),
        "energy_std": float(arrays.E.std()),
        "force_rms": float(np.sqrt(np.mean(arrays.F**2))),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def split_indices(
    n_frames: int,
    train_size: int,
    val_size: int,
    test_size: int,
    stride: int = 1,
) -> dict[str, np.ndarray]:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    indices = np.arange(0, n_frames, stride, dtype=np.int64)
    total = train_size + val_size + test_size
    if total > len(indices):
        raise ValueError(
            f"Requested {total} frames after stride={stride}, but dataset only has {len(indices)}"
        )
    train_end = train_size
    val_end = train_end + val_size
    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end : val_end + test_size],
    }


class MD17Dataset(Dataset):
    def __init__(self, arrays: MD17Arrays, indices: np.ndarray):
        self.z = torch.as_tensor(arrays.z, dtype=torch.long)
        self.R = torch.as_tensor(arrays.R[indices], dtype=torch.float32)
        self.E = torch.as_tensor(arrays.E[indices], dtype=torch.float32)
        self.F = torch.as_tensor(arrays.F[indices], dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.R.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "z": self.z,
            "R": self.R[idx],
            "E": self.E[idx],
            "F": self.F[idx],
            "frame": torch.tensor(idx, dtype=torch.long),
        }


class TemporalMD17Dataset(Dataset):
    def __init__(self, arrays: MD17Arrays, indices: np.ndarray, delta: int):
        if delta < 0:
            raise ValueError("delta must be >= 0")
        valid = indices[indices + delta < arrays.R.shape[0]]
        if len(valid) == 0:
            raise ValueError("No valid temporal pairs for requested indices and delta")
        self.z = torch.as_tensor(arrays.z, dtype=torch.long)
        self.R_context = torch.as_tensor(arrays.R[valid], dtype=torch.float32)
        self.R_target = torch.as_tensor(arrays.R[valid + delta], dtype=torch.float32)
        self.indices = torch.as_tensor(valid, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.R_context.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "z": self.z,
            "R_context": self.R_context[idx],
            "R_target": self.R_target[idx],
            "frame": self.indices[idx],
        }


def collate_static(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    z = batch[0]["z"]
    return {
        "z": z.unsqueeze(0).expand(len(batch), -1).contiguous(),
        "R": torch.stack([item["R"] for item in batch], dim=0),
        "E": torch.stack([item["E"] for item in batch], dim=0),
        "F": torch.stack([item["F"] for item in batch], dim=0),
        "frame": torch.stack([item["frame"] for item in batch], dim=0),
    }


def collate_temporal(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    z = batch[0]["z"]
    return {
        "z": z.unsqueeze(0).expand(len(batch), -1).contiguous(),
        "R_context": torch.stack([item["R_context"] for item in batch], dim=0),
        "R_target": torch.stack([item["R_target"] for item in batch], dim=0),
        "frame": torch.stack([item["frame"] for item in batch], dim=0),
    }
