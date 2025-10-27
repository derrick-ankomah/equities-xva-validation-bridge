# 02 — Model Uncertainty: Conformal + Bootstrap

**Goal**: Quantify **local & global uncertainty** around an equities forecaster (returns or realized volatility).
- **Conformal** (distribution-free) **intervals** with user-specified miscoverage.
- **Bootstrap retraining** to capture **parameter/data** uncertainty.
- Combine both to set **position size** or **limits**.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r python/requirements.txt
(cd cpp && python -m build && pip install dist/*.whl)
python python/run_uncertainty.py --alpha 0.1 --B 100
```

Data note: script includes a `--use-yfinance` flag to download SPY data; otherwise it uses synthetic.
