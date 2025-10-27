# 03 — Offset Residual Learner (Global & Local Weakness)

**Goal**: Identify **systematic weaknesses** in a base model via an **offset learner** trained on residuals.
- **Global**: average residual by regime (e.g., moneyness, tenor, sector).
- **Local**: shallow model (depth<=2) on residuals to expose *where* base fails.

**Retail bridge**: Mirrors offset-residual tests used to find **weak segments** in PD/scorecards; here used to detect **vol surface/skew biases** or **return-forecast drift**.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r python/requirements.txt
(cd cpp && python -m build && pip install dist/*.whl)
python python/run_offset.py --depth 2
```
