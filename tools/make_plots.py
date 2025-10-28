import os, json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).resolve().parent.parent
R = root / "reports"
(R/"robustness").mkdir(parents=True, exist_ok=True)
(R/"uncertainty").mkdir(parents=True, exist_ok=True)
(R/"offset").mkdir(parents=True, exist_ok=True)

# Robustness
p = R/"robustness"/"summary.json"
if p.exists():
    s = json.loads(p.read_text())
    vals = [s.get("local_mean",0), s.get("local_p95",0), s.get("global_mean",0), s.get("global_p95",0)]
    labels = ["local_mean","local_p95","global_mean","global_p95"]
    plt.figure(); plt.bar(labels, vals); plt.title(f"Robustness drift ({s.get('result','?')})"); plt.xticks(rotation=20)
    plt.tight_layout(); plt.savefig(R/"robustness"/"drift.png", dpi=160); plt.close()

# Uncertainty
p = R/"uncertainty"/"summary.json"
if p.exists():
    s = json.loads(p.read_text())
    q = s.get("conformal_radius")
    if q is not None:
        # simple band visual mock: constant band around an index
        import numpy as np
        x = np.arange(100); pred = np.sin(x/12.0)
        plt.figure(); plt.plot(x, pred, label="pred")
        plt.fill_between(x, pred - q, pred + q, alpha=0.3, label=f"conformal ±{q:.3f}")
        plt.legend(); plt.title("Uncertainty bands (demo)")
        plt.tight_layout(); plt.savefig(R/"uncertainty"/"bands.png", dpi=160); plt.close()

# Offset (bar of residual group means if available in log; otherwise just R^2s)
p = R/"offset"/"summary.json"
if p.exists():
    s = json.loads(p.read_text())
    base = s.get("base_r2"); fixed = s.get("fixed_r2")
    if base is not None and fixed is not None:
        plt.figure(); plt.bar(["base_R2","fixed_R2"], [base, fixed]); plt.title("Offset: R² before/after")
        plt.tight_layout(); plt.savefig(R/"offset"/"r2.png", dpi=160); plt.close()
