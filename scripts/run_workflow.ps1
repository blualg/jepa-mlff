param(
    [switch]$Quick,
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PythonArgs)
    python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "python $($PythonArgs -join ' ') failed with exit code $LASTEXITCODE"
    }
}

Invoke-Python --version
Invoke-Python -m pip install -r requirements.txt

Invoke-Python scripts/download_md17.py

if ($Quick) {
    $TrainSize = 64
    $ValSize = 16
    $TestSize = 16
    $Epochs = 1
    $Batch = 8
    $Hidden = 48
    $Layers = 2
} else {
    $TrainSize = 2000
    $ValSize = 500
    $TestSize = 500
    $Epochs = 10
    $Batch = 16
    $Hidden = 128
    $Layers = 4
}

$dataArgs = @(
    "--train-size", $TrainSize,
    "--val-size", $ValSize,
    "--test-size", $TestSize,
    "--batch-size", $Batch,
    "--device", $Device
)

$trainArgs = @(
    "--train-size", $TrainSize,
    "--val-size", $ValSize,
    "--test-size", $TestSize,
    "--batch-size", $Batch,
    "--hidden-dim", $Hidden,
    "--layers", $Layers,
    "--device", $Device
)

Invoke-Python scripts/pretrain_jepa.py @trainArgs --epochs $Epochs --out checkpoints/jepa_pretrained.pt
Invoke-Python scripts/finetune_forcefield.py @trainArgs --epochs $Epochs --out checkpoints/forcefield_scratch.pt --metrics results/finetune_scratch_metrics.json
Invoke-Python scripts/finetune_forcefield.py @trainArgs --epochs $Epochs --pretrained checkpoints/jepa_pretrained.pt --out checkpoints/forcefield_jepa.pt --metrics results/finetune_jepa_metrics.json
Invoke-Python scripts/evaluate.py @dataArgs --scratch checkpoints/forcefield_scratch.pt --jepa checkpoints/forcefield_jepa.pt

Write-Host "Workflow complete. See results/metrics.json and results/*.png"
