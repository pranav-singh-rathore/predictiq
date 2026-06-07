#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
cd src
python train.py --epochs 25 --output ../artifacts
python evaluate.py --model ../artifacts/model.pt --output ../artifacts
python quantize.py --model ../artifacts/model.pt --output ../artifacts
