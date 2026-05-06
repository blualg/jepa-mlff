from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jepa_md17_mlff.data import MD17Dataset, collate_static, load_md17_npz, split_indices
from jepa_md17_mlff.model import AtomisticEncoder, ForceField
from jepa_md17_mlff.train_utils import MetricAverager, load_checkpoint, pick_device, save_json


def infer_model_config(ckpt: dict) -> dict:
    config = ckpt.get("config", {})
    return {
        "hidden_dim": int(config.get("hidden_dim", 128)),
        "layers": int(config.get("layers", 4)),
        "rbf": int(config.get("rbf", 32)),
        "cutoff": float(config.get("cutoff", 5.0)),
    }


def load_model(path: Path, device: torch.device) -> tuple[ForceField, dict]:
    ckpt = load_checkpoint(path, device)
    cfg = infer_model_config(ckpt)
    model = ForceField(
        AtomisticEncoder(
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["layers"],
            n_rbf=cfg["rbf"],
            cutoff=cfg["cutoff"],
        )
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt


def evaluate_model(model: ForceField, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    avg = MetricAverager()
    all_e_true = []
    all_e_pred = []
    all_f_true = []
    all_f_pred = []
    for batch in loader:
        z = batch["z"].to(device)
        R = batch["R"].to(device)
        E_true = batch["E"].to(device)
        F_true = batch["F"].to(device)
        with torch.enable_grad():
            E_pred, F_pred = model.energy_and_forces(z, R)
        e_mae = F.l1_loss(E_pred, E_true)
        f_mae = F.l1_loss(F_pred, F_true)
        f_rmse = torch.sqrt(F.mse_loss(F_pred, F_true))
        avg.update(
            {"energy_mae": e_mae.item(), "force_mae": f_mae.item(), "force_rmse": f_rmse.item()},
            z.shape[0],
        )
        all_e_true.append(E_true.detach().cpu().numpy())
        all_e_pred.append(E_pred.detach().cpu().numpy())
        all_f_true.append(F_true.detach().cpu().numpy().reshape(-1))
        all_f_pred.append(F_pred.detach().cpu().numpy().reshape(-1))
    raw = {
        "E_true": np.concatenate(all_e_true),
        "E_pred": np.concatenate(all_e_pred),
        "F_true": np.concatenate(all_f_true),
        "F_pred": np.concatenate(all_f_pred),
    }
    metrics = avg.compute()
    metrics["rollout_drift_proxy"] = float(np.mean(np.abs(raw["F_pred"] - raw["F_true"])))
    return metrics, raw


def plot_force_scatter(raw_by_name: dict[str, dict[str, np.ndarray]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(raw_by_name), figsize=(5 * len(raw_by_name), 4), squeeze=False)
    for ax, (name, raw) in zip(axes[0], raw_by_name.items()):
        n = min(5000, raw["F_true"].shape[0])
        idx = np.linspace(0, raw["F_true"].shape[0] - 1, n).astype(int)
        ax.scatter(raw["F_true"][idx], raw["F_pred"][idx], s=3, alpha=0.35)
        lo = float(min(raw["F_true"][idx].min(), raw["F_pred"][idx].min()))
        hi = float(max(raw["F_true"][idx].max(), raw["F_pred"][idx].max()))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("DFT force")
        ax.set_ylabel("Predicted force")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_loss_curves(checkpoints: dict[str, dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, ckpt in checkpoints.items():
        hist = ckpt.get("history", [])
        if not hist:
            continue
        epochs = [row["epoch"] for row in hist]
        vals = [row.get("val_force_rmse", np.nan) for row in hist]
        ax.plot(epochs, vals, marker="o", label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation force RMSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MD17 force-field checkpoints.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "md17_ethanol.npz")
    parser.add_argument("--jepa", type=Path, default=ROOT / "checkpoints" / "forcefield_jepa.pt")
    parser.add_argument("--scratch", type=Path, default=ROOT / "checkpoints" / "forcefield_scratch.pt")
    parser.add_argument("--metrics", type=Path, default=ROOT / "results" / "metrics.json")
    parser.add_argument("--force-plot", type=Path, default=ROOT / "results" / "force_scatter.png")
    parser.add_argument("--loss-plot", type=Path, default=ROOT / "results" / "loss_curves.png")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = pick_device(args.device)
    arrays = load_md17_npz(args.data)
    splits = split_indices(arrays.R.shape[0], args.train_size, args.val_size, args.test_size, args.stride)
    test_loader = DataLoader(MD17Dataset(arrays, splits["test"]), batch_size=args.batch_size, shuffle=False, collate_fn=collate_static)

    metrics = {}
    raws = {}
    ckpts = {}
    for name, path in [("scratch", args.scratch), ("jepa", args.jepa)]:
        if not path.exists():
            print(f"Skipping missing checkpoint: {path}")
            continue
        model, ckpt = load_model(path, device)
        ckpts[name] = ckpt
        metrics[name], raws[name] = evaluate_model(model, test_loader, device)
        print(name, json.dumps(metrics[name], indent=2))

    save_json(args.metrics, metrics)
    if raws:
        plot_force_scatter(raws, args.force_plot)
    if ckpts:
        plot_loss_curves(ckpts, args.loss_plot)
    print(f"Wrote {args.metrics}")


if __name__ == "__main__":
    main()
