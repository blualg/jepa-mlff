# JEPA-MLFF

Research MVP for JEPA-style self-supervised pretraining of a neural molecular force field on real MD17/rMD17-style molecular dynamics data.

The workflow uses ethanol by default because it is a standard small-molecule DFT trajectory benchmark with atomic numbers, coordinates, energies, and forces. The downloader tries canonical sGDML/MD17 URLs such as `http://www.quantum-machine.org/gdml/data/npz/md17_ethanol.npz` and compatible fallback names. The data loader accepts common MD17/rMD17 `.npz` key variants: `z/Z`, `R`, `E`, and `F`.

## What It Builds

- A compact SchNet-like continuous-filter encoder in plain PyTorch.
- JEPA pretraining with online encoder, EMA target encoder, masked atoms, and latent prediction.
- Conservative force-field fine-tuning where forces are `-dE/dR`.
- Scratch-vs-JEPA evaluation with energy MAE, force MAE/RMSE, a simple force-drift proxy, and plots.

## Quick Start

From PowerShell:

```powershell
cd path\to\jepa-mlff
.\scripts\run_workflow.ps1 -Quick
```

The quick workflow installs requirements, downloads data, trains tiny models for one epoch, and writes:

- `checkpoints/jepa_pretrained.pt`
- `checkpoints/forcefield_scratch.pt`
- `checkpoints/forcefield_jepa.pt`
- `results/metrics.json`
- `results/loss_curves.png`
- `results/force_scatter.png`

For a longer CPU/small-GPU run:

```powershell
.\scripts\run_workflow.ps1
```

## Manual Commands

```powershell
python -m pip install -r requirements.txt
python scripts/download_md17.py
python scripts/pretrain_jepa.py --epochs 10
python scripts/finetune_forcefield.py --epochs 20 --out checkpoints/forcefield_scratch.pt --metrics results/finetune_scratch_metrics.json
python scripts/finetune_forcefield.py --epochs 20 --pretrained checkpoints/jepa_pretrained.pt --out checkpoints/forcefield_jepa.pt
python scripts/evaluate.py
```

## Notes

This is a research scaffold, not a benchmark-optimized MLIP package. The model is intentionally small and dependency-light so the whole idea can be inspected, modified, and run on modest hardware. For serious accuracy, scale the hidden size, train longer, use larger rMD17 splits, and compare against established equivariant MLIP libraries such as NequIP or MACE.

If you cite this repository in a manuscript, use the public repository URL in the manuscript's Code availability section.

