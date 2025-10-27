import argparse, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import fastops

def make_synth(n=1500, d=12, seed=1):
    rng=np.random.default_rng(seed)
    X=rng.normal(size=(n,d))
    beta=np.linspace(0.8, -0.1, d)
    y=X@beta + 0.5*rng.standard_normal(n)
    return X,y

def conformal_interval(y_true, y_pred, alpha=0.1):
    # absolute residual conformity
    resid=np.abs(y_true - y_pred)
    q=np.quantile(resid, 1-alpha)
    return q

def bootstrap_bands(X, y, model, B=100, alpha=0.1, seed=7):
    n=len(y)
    idx = fastops.bootstrap_indices(n, B, seed=seed)
    preds=[]
    for b in range(B):
        ii = np.array(idx[b], dtype=int)
        m = RandomForestRegressor(n_estimators=200, random_state=b)
        m.fit(X[ii], y[ii])
        preds.append(m.predict(X))
    P = np.vstack(preds)  # B x n
    lo = np.quantile(P, alpha/2, axis=0)
    hi = np.quantile(P, 1-alpha/2, axis=0)
    return lo, hi

def maybe_load_yfinance():
    try:
        import yfinance as yf
        df = yf.download("SPY", period="5y", interval="1d", auto_adjust=True, progress=False)
        df["ret"]=df["Close"].pct_change()
        df["rv"]= (np.log(df["Close"]).diff().rolling(5).std()*np.sqrt(252)).shift(-1)
        df=df.dropna()
        X = np.stack([df["ret"].rolling(k).mean().shift(1).fillna(0).to_numpy() for k in [2,5,10,20,60]] ,axis=1)
        y = df["rv"].to_numpy()
        return X,y
    except Exception as e:
        return None

def main(args):
    if args.use_yfinance:
        data = maybe_load_yfinance()
    else:
        data=None
    if data is None:
        X,y = make_synth()
    else:
        X,y = data

    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.25, random_state=42)
    model = RandomForestRegressor(n_estimators=400, random_state=42)
    model.fit(Xtr,ytr)
    yhat = model.predict(Xte)

    # Conformal band (calibrated on a holdout)
    q = conformal_interval(yte, yhat, alpha=args.alpha)

    # Bootstrap predictive band around in-sample fit
    lo, hi = bootstrap_bands(Xtr, ytr, model, B=args.B, alpha=args.alpha)
    print(f"Conformal radius q={q:.4f}; Bootstrap LO/HI percentiles computed.")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--B", type=int, default=100)
    ap.add_argument("--use-yfinance", action="store_true")
    args=ap.parse_args()
    main(args)
