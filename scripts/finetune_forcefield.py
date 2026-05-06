from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jepa_md17_mlff.data import MD17Dataset, collate_static, load_md17_npz, split_indices
from jepa_md17_mlff.model import AtomisticEncoder, ForceField
from jepa_md17_mlff.train_utils import MetricAverager, load_checkpoint, pick_device, save_json, set_seed


def make_model(args, device: torch.device) -> ForceField:
    encoder = AtomisticEncoder(hidden_dim=args.hidden_dim, n_layers=args.layers, n_rbf=args.rbf, cutoff=args.cutoff)
    model = ForceField(encoder).to(device)
    if args.pretrained:
        ckpt = load_checkpoint(args.pretrained, device)
        model.encoder.load_state_dict(ckpt["encoder"], strict=True)
        print(f"Loaded JEPA encoder from {args.pretrained}")
    return model


def batch_loss(model: ForceField, batch: dict[str, torch.Tensor], args, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    z = batch["z"].to(device)
    R = batch["R"].to(device)
    E_true = batch["E"].to(device)
    F_true = batch["F"].to(device)
    E_pred, F_pred = model.energy_and_forces(z, R)
    e_loss = F.l1_loss(E_pred, E_true)
    f_loss = F.mse_loss(F_pred, F_true)
    loss = args.energy_weight * e_loss + args.force_weight * f_loss
    metrics = {
        "loss": loss.item(),
        "energy_mae": e_loss.item(),
        "force_rmse": torch.sqrt(f_loss).item(),
        "force_mae": F.l1_loss(F_pred, F_true).item(),
    }
    return loss, metrics


@torch.no_grad()
def evaluate(model: ForceField, loader: DataLoader, args, device: torch.device) -> dict[str, float]:
    model.eval()
    avg = MetricAverager()
    for batch in loader:
        z = batch["z"].to(device)
        R = batch["R"].to(device)
        E_true = batch["E"].to(device)
        F_true = batch["F"].to(device)
        with torch.enable_grad():
            E_pred, F_pred = model.energy_and_forces(z, R)
        e_loss = F.l1_loss(E_pred, E_true)
        f_mse = F.mse_loss(F_pred, F_true)
        avg.update(
            {
                "energy_mae": e_loss.item(),
                "force_rmse": torch.sqrt(f_mse).item(),
                "force_mae": F.l1_loss(F_pred, F_true).item(),
            },
            z.shape[0],
        )
    return avg.compute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune conservative force field on MD17.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "md17_ethanol.npz")
    parser.add_argument("--pretrained", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "forcefield_jepa.pt")
    parser.add_argument("--metrics", type=Path, default=ROOT / "results" / "finetune_jepa_metrics.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--rbf", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--energy-weight", type=float, default=0.05)
    parser.add_argument("--force-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    arrays = load_md17_npz(args.data)
    splits = split_indices(arrays.R.shape[0], args.train_size, args.val_size, args.test_size, args.stride)
    train_ds = MD17Dataset(arrays, splits["train"])
    val_ds = MD17Dataset(arrays, splits["val"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_static)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_static)

    model = make_model(args, device)
    model.energy_offset.fill_(float(train_ds.E.mean()))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    history = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        avg = MetricAverager()
        progress = tqdm(train_loader, desc=f"FF epoch {epoch}/{args.epochs}")
        for batch in progress:
            loss, metrics = batch_loss(model, batch, args, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            avg.update(metrics, batch["R"].shape[0])
            progress.set_postfix(force_rmse=f"{metrics['force_rmse']:.3f}")
        val = evaluate(model, val_loader, args, device)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in avg.compute().items()}, **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print(row)
        score = val["force_rmse"]
        if score < best_val:
            best_val = score
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "config": vars(args), "history": history}, args.out)

    save_json(args.metrics, {"best_val_force_rmse": best_val, "history": history, "checkpoint": str(args.out)})


if __name__ == "__main__":
    main()
