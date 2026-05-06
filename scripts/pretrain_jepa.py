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

from jepa_md17_mlff.data import TemporalMD17Dataset, collate_temporal, load_md17_npz, split_indices
from jepa_md17_mlff.model import AtomisticEncoder, JEPAPredictor
from jepa_md17_mlff.train_utils import (
    MetricAverager,
    pick_device,
    random_atom_mask,
    save_json,
    set_seed,
    update_ema,
)


def evaluate(loader: DataLoader, online: AtomisticEncoder, target: AtomisticEncoder, predictor: JEPAPredictor, args, device) -> float:
    online.eval()
    target.eval()
    predictor.eval()
    avg = MetricAverager()
    with torch.no_grad():
        for batch in loader:
            z = batch["z"].to(device)
            R_context = batch["R_context"].to(device)
            R_target = batch["R_target"].to(device)
            mask = random_atom_mask(z.shape[0], z.shape[1], args.mask_ratio, device)
            pred = predictor(online(z, R_context, atom_mask=mask))
            target_h = target(z, R_target)
            loss = F.mse_loss(pred[mask], target_h[mask])
            avg.update({"loss": loss.item()}, z.shape[0])
    return avg.compute()["loss"]


def main() -> None:
    parser = argparse.ArgumentParser(description="JEPA pretraining on MD17 trajectories.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "md17_ethanol.npz")
    parser.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "jepa_pretrained.pt")
    parser.add_argument("--metrics", type=Path, default=ROOT / "results" / "pretrain_metrics.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--delta", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--rbf", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--mask-ratio", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ema", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    arrays = load_md17_npz(args.data)
    splits = split_indices(arrays.R.shape[0], args.train_size, args.val_size, args.test_size, args.stride)
    train_ds = TemporalMD17Dataset(arrays, splits["train"], delta=args.delta)
    val_ds = TemporalMD17Dataset(arrays, splits["val"], delta=args.delta)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_temporal)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_temporal)

    online = AtomisticEncoder(hidden_dim=args.hidden_dim, n_layers=args.layers, n_rbf=args.rbf, cutoff=args.cutoff).to(device)
    target = AtomisticEncoder(hidden_dim=args.hidden_dim, n_layers=args.layers, n_rbf=args.rbf, cutoff=args.cutoff).to(device)
    target.load_state_dict(online.state_dict())
    target.requires_grad_(False)
    predictor = JEPAPredictor(hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.AdamW([*online.parameters(), *predictor.parameters()], lr=args.lr, weight_decay=1e-5)

    history = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        online.train()
        predictor.train()
        avg = MetricAverager()
        progress = tqdm(train_loader, desc=f"JEPA epoch {epoch}/{args.epochs}")
        for batch in progress:
            z = batch["z"].to(device)
            R_context = batch["R_context"].to(device)
            R_target = batch["R_target"].to(device)
            mask = random_atom_mask(z.shape[0], z.shape[1], args.mask_ratio, device)
            pred = predictor(online(z, R_context, atom_mask=mask))
            with torch.no_grad():
                target_h = target(z, R_target)
            loss = F.mse_loss(pred[mask], target_h[mask])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            update_ema(online, target, args.ema)
            avg.update({"train_loss": loss.item()}, z.shape[0])
            progress.set_postfix(loss=f"{loss.item():.4f}")
        val_loss = evaluate(val_loader, online, target, predictor, args, device)
        row = {"epoch": epoch, **avg.compute(), "val_loss": val_loss}
        history.append(row)
        print(row)
        if val_loss < best_val:
            best_val = val_loss
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "encoder": online.state_dict(),
                    "target_encoder": target.state_dict(),
                    "predictor": predictor.state_dict(),
                    "config": vars(args),
                    "history": history,
                },
                args.out,
            )

    save_json(args.metrics, {"best_val_loss": best_val, "history": history, "checkpoint": str(args.out)})


if __name__ == "__main__":
    main()
