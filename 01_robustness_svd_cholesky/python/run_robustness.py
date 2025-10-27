import argparse, numpy as np, pandas as pd
from numpy.linalg import svd, cholesky
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

import fastops  # from cpp build

def make_synthetic(n=2000, d=20, seed=7):
    rng = np.random.default_rng(seed)
    Sigma = np.diag(np.linspace(1.0, 0.2, d))
    L = cholesky(Sigma)
    X = rng.normal(size=(n,d)) @ L.T
    beta = np.linspace(0.5, -0.3, d)
    y = X @ beta + 0.1 * rng.standard_normal(n)
    return X, y, L, beta

def local_svd_perturb(x, eps=0.02, rank=3):
    U, s, Vt = svd(np.cov(x.T), full_matrices=False)
    x2 = np.vstack([fastops.svd_perturb(U, s, xi, eps, rank) for xi in x])
    return x2

def global_chol_draw(mu, L, n_paths=1000, seed=123):
    return fastops.chol_draw(L, mu, n_paths, seed)

def main(args):
    X, y, L, beta = make_synthetic(args.n, args.d, seed=7)
    model = Ridge(alpha=1.0).fit(X, y)
    yhat = model.predict(X)

    # Local robustness: SVD perturb
    Xp = local_svd_perturb(X, eps=args.eps, rank=args.rank)
    yhat_p = model.predict(Xp)
    drift = np.abs(yhat_p - yhat)
    print(f"Local drift@eps={args.eps}: mean={drift.mean():.4f}, p95={np.percentile(drift,95):.4f}")

    # Global robustness: simulate factor shocks
    mu = np.zeros(X.shape[1])
    Xg = global_chol_draw(mu, L, n_paths=X.shape[0], seed=11)
    yhat_g = model.predict(Xg)
    g_drift = np.abs(yhat_g - yhat)
    print(f"Global drift: mean={g_drift.mean():.4f}, p95={np.percentile(g_drift,95):.4f}")

    # Simple gate
    passed = (drift.mean() < 0.05) and (g_drift.mean() < 0.05)
    print("ROBUSTNESS PASS" if passed else "ROBUSTNESS FAIL")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--d", type=int, default=20)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--rank", type=int, default=3)
    args = ap.parse_args()
    main(args)
