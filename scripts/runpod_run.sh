#!/usr/bin/env bash
# One command on a RunPod PyTorch pod (A100/H100, PyTorch template). About 15-20 minutes in total.
#   bash <(curl -fsSL https://raw.githubusercontent.com/nsquaredzz/certified-skip/claude/great-darwin-duxo8h/scripts/runpod_run.sh)
# or, with the repository already cloned:   bash scripts/runpod_run.sh
# MINUTES=10 sets the training budget; the evaluation takes a few minutes more. Results: ~/certified-skip/longvid_results.tgz
set -euo pipefail
REPO=${REPO:-https://github.com/nsquaredzz/certified-skip.git}; BRANCH=${BRANCH:-claude/great-darwin-duxo8h}
cd /workspace 2>/dev/null || cd "$HOME"
[ -d certified-skip ] || git clone --depth 1 -b "$BRANCH" "$REPO"
cd certified-skip && git pull -q --ff-only || true
pip install -q numpy pillow pytest
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -m pytest -q tests/test_longvid.py
echo "--- preflight: the real model for 25 seconds, so memory and speed are known before the long run"
python3 scripts/longvid_gpu.py --minutes 0.4 --eval-episodes 2 --eval-batch 2 --eval-frames 20 --steps 2 --eps 0.1 --out /tmp/preflight 2>&1 | grep -E "device|step|trained|full |took"
echo "--- main run"
python3 scripts/longvid_gpu.py --minutes "${MINUTES:-10}" --resume --out out/longvid_gpu
tar czf longvid_results.tgz out/longvid_gpu.txt out/longvid_gpu.json
echo "done: download longvid_results.tgz (Jupyter file browser, or: runpodctl send longvid_results.tgz), then TERMINATE the pod"
