#!/usr/bin/env bash
# Create / update the isolated conda env `rsdbench` with CUDA PyTorch (GPU only).
# Does NOT touch other users' environments or shared conda/pip config.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# Only used for this script's pip commands — not written into the env permanently.
PIP_MIRROR=(-i https://pypi.tuna.tsinghua.edu.cn/simple)

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found."
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx rsdbench; then
  echo "[setup] conda env rsdbench already exists"
else
  echo "[setup] creating conda env rsdbench (python 3.10, conda-forge only)"
  # --override-channels avoids Anaconda.com ToS on default pkgs/main|r channels
  conda create -y -n rsdbench --override-channels -c conda-forge python=3.10 pip
fi

conda activate rsdbench

echo "[setup] installing CUDA PyTorch + torchvision (GPU; not cpu-only)"
python -m pip install "${PIP_MIRROR[@]}" --default-timeout=300 torch torchvision
python -m pip install "${PIP_MIRROR[@]}" --default-timeout=300 -r "${ROOT}/requirements.txt"

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("torch.version.cuda", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA PyTorch not usable. Refuse CPU-only setup for RSDBench.")
print("device0", torch.cuda.get_device_name(0))
x = torch.randn(2, 3, device="cuda")
print("gpu_smoke_ok", float((x @ x.T).sum().item()))
PY

echo "[setup] done. Activate with: conda activate rsdbench"
