# 01 — Robustness via SVD & Cholesky

**Goal**: Evaluate *local & global robustness* of an equities forecaster (e.g., next-day realized volatility or short-horizon return signal) by perturbing:
- **Inputs (features)** along **low-rank SVD directions**;
- **Risk-factor draws** using **Cholesky** of a target covariance.

**Reports**:
- Sensitivity of predictions to structured perturbations,
- Stability curves (prediction drift vs eps),
- Pass/Fail gates for production (max drift %, stability KS, etc.).

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r python/requirements.txt
(cd cpp && python -m build && pip install dist/*.whl)
python python/run_robustness.py --n 2000 --d 20 --eps 0.02 --rank 3
```

See `notebooks/robustness_demo.ipynb` for figures.
