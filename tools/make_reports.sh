#!/usr/bin/env bash
set -euo pipefail

# Ensure we're in repo root
cd "$(git rev-parse --show-toplevel)"

# Run each pipeline and capture stdout to logs
python 01_robustness_svd_cholesky/python/run_robustness.py \
  | tee reports/robustness/run.log

python 02_uncertainty_conformal_bootstrap/python/run_uncertainty.py --use-yfinance \
  | tee reports/uncertainty/run.log

# Offset may fail on some setups -> don't stop the whole script
python 03_offset_model_weakness/python/run_offset.py \
  | tee reports/offset/run.log || true
