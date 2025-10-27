# equities-xva-validation-bridge
=======
# Equities & XVA – Validation-to-FO Bridge

This monorepo contains **three production-style, research-grade projects** that map your **retail credit scoring model validation** toolkit directly into **Front-Office Equities/XVA** problems.

> **Core through-line**: the same validation primitives you used in retail (SVD/Cholesky perturbations, conformal + bootstrap uncertainty, offset residual learners) are *powerful and transferable* for market risk, trading, and equity-derivatives pricing.

## Why this repo fits the job
- **FO Quant fit**: Implements *forecasting, optimization, and risk mitigation* in equities; shows *model design, code quality, C++ performance, and docs* aligned with Agile SDLC.
- **Cross-asset platform mindset**: Each project uses shared patterns (data, features, evaluation, CI-ready layout), making integration into a *holistic quant platform* straightforward.
- **C++ acceleration**: Hot loops (linear algebra, bootstrapping, residual stats) are implemented with **pybind11** C++ extensions. This improves:
  - **Throughput & latency** on large recalcs (e.g., intraday risk or batch reprice),
  - **Determinism** & **memory control**, enabling desk-style SLAs,
  - **Numerical robustness** (e.g., pivoted Cholesky, SVD truncation for stability).

## Projects
1. **01_robustness_svd_cholesky** — *Local & global robustness via SVD/Cholesky perturbations*  
   - Equities mapping: perturb **feature vectors and risk-factor covariances** to stress **signal stability** for returns/vol forecasts or Greeks approximators.
   - Retail bridge: mirrors **application/account management stress tests** where covariances/inputs are nudged to test **model stability**.

2. **02_uncertainty_conformal_bootstrap** — *Local & global model uncertainty*  
   - Conformal prediction bands + **bootstrap retraining** for volatility/return forecasts.  
   - Equities mapping: **intervals around forecasts** to size positions and **set stop-loss/limits** with quantile guarantees.  
   - Retail bridge: analogous to **PD/score uncertainty** and **bands on risk estimates**.

3. **03_offset_model_weakness** — *Local & global performance weakness via offset learners*  
   - Train an **offset model** on residuals to reveal **systematic bias** by regime (moneyness, tenor, sector, liquidity).  
   - Equities/XVA mapping: diagnose **where the base model misses** (e.g., vol surface wings, short-dated skew), then fix with **targeted features/constraints**.

Each project includes:
- **`cpp/`**: pybind11 modules (`fastops`) for linear-algebra kernels, bootstrap, and residual stats.
- **`python/`**: sklearn/LightGBM workflows, unit tests, and CLI scripts.
- **Notebooks**: tiny starters to reproduce key figures quickly.
- **`README.md`**: FO narrative, how-to-run, and acceptance criteria.

## Build & Run (macOS)
```bash
# Prereqs
xcode-select --install            # compilers
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip wheel build pybind11 scikit-build-core numpy pandas scipy scikit-learn lightgbm matplotlib

# Build C++ modules for each project
for d in 01_robustness_svd_cholesky 02_uncertainty_conformal_bootstrap 03_offset_model_weakness; do
  (cd $d/cpp && python -m build && pip install dist/*.whl)
done
```

## Why *these* methods signal original thinking in Equities/XVA

- **SVD/Cholesky perturbations**: In retail, you perturbed inputs to test **local/global stability**. In equities/XVA, perturb **risk factor covariance** or **feature embeddings** to emulate **microstructure noise**, **regime shifts**, or **vol surface shocks**. SVD helps isolate **low-energy/ill-conditioned directions**; Cholesky supports **structured draws** from a target covariance.

- **Conformal + Bootstrap**: Conformal provides **model-agnostic coverage** for **forecast intervals**; bootstrap retraining quantifies **parameter & data uncertainty**. Together they provide **decision-grade bands** for **position sizing** and **limit setting**—directly actionable for trading and risk.

- **Offset residual learners**: Your retail offset test to find **global/local weakness** becomes a **bias detector** for FO models (e.g., underpricing OTM puts, short-tenor smile). The offset learner maps *where* and *why* the base model errs and offers **low-complexity fixes** that preserve interpretability.

---

## Repo layout
```
equities-xva-validation-bridge/
  01_robustness_svd_cholesky/
  02_uncertainty_conformal_bootstrap/
  03_offset_model_weakness/
  README.md
```

## License
MIT
>>>>>>> 0affc5b (Initial commit: robustness, uncertainty, offset projects with C++ fastops)
