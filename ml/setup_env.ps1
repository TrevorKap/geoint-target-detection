# ─────────────────────────────────────────────────────────────────────────────
# setup_env.ps1 — create the Python venv and install the ML stack (Windows).
#
#   Run from the project root:  powershell -ExecutionPolicy Bypass -File ml\setup_env.ps1
#
# Installs CUDA-enabled PyTorch (cu124) for the NVIDIA RTX 3070 Ti, then the rest
# of ml/requirements.txt. Re-runnable: skips venv creation if it already exists.
# ─────────────────────────────────────────────────────────────────────────────
$ErrorActionPreference = 'Stop'

$Root   = Split-Path -Parent $PSScriptRoot
$Venv   = Join-Path $Root '.venv'
$Py     = Join-Path $Venv 'Scripts\python.exe'

if (-not (Test-Path $Venv)) {
    Write-Host '==> Creating virtual environment (.venv)…'
    py -3.12 -m venv $Venv
} else {
    Write-Host '==> .venv already exists, reusing.'
}

Write-Host '==> Upgrading pip…'
& $Py -m pip install --upgrade pip --quiet

Write-Host '==> Installing CUDA PyTorch (cu124)…'
& $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Write-Host '==> Installing ML requirements…'
& $Py -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')

Write-Host '==> Verifying CUDA availability…'
& $Py -c "import torch; print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

Write-Host ''
Write-Host 'Done. Activate with:  .\.venv\Scripts\Activate.ps1'
