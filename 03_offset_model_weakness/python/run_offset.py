import argparse, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
import fastops

def synth_equity_option(n=5000, seed=0):
    rng=np.random.default_rng(seed)
    # Toy features: moneyness, tenor, rate, vol level
    mny = rng.uniform(0.8,1.2,n)
    ten = rng.uniform(7,90,n)
    r   = rng.uniform(0.0,0.05,n)
    vol = rng.uniform(0.1,0.6,n)
    X = np.column_stack([mny,ten,r,vol])
    # Base price model (unknown true), add regime bias
    y = (1.5*(mny-1)**2 + 0.01*ten + 5*r + 2*vol) + 0.05*rng.standard_normal(n)
    # Inject bias: underprice deep OTM (mny<0.9) short tenor
    y -= 0.08*((mny<0.9)&(ten<20))
    # Group ids for global stats (bin moneyness)
    gid = (np.floor((mny-0.8)/0.05)).astype(int)
    return X,y,gid

def main(args):
    X,y,gid = synth_equity_option()
    Xtr,Xte,ytr,yte, gtr, gte = train_test_split(X,y,gid,test_size=0.25,random_state=42)

    # Baseline model (over-flexible to mimic "black-box")
    base = lgb.LGBMRegressor(n_estimators=600, max_depth=-1, learning_rate=0.05, subsample=0.7, colsample_bytree=0.8, random_state=1)
    base.fit(Xtr,ytr)
    yhat = base.predict(Xte)
    print("Base R2:", r2_score(yte,yhat))

    # Residuals
    res = yte - yhat

    # Global weakness: residual mean per group via C++ kernel
    ids, mean_res = fastops.residual_group_mean(yte, yhat, gte.astype(np.int64))
    print("Top negative mean residual groups (most underpriced):")
    order = np.argsort(mean_res)
    for idx in order[:5]:
        print(f"  group={ids[idx]} mean_res={mean_res[idx]:.4f}")

    # Local weakness: shallow offset learner on residuals
    off = DecisionTreeRegressor(max_depth=args.depth, random_state=0)
    off.fit(Xte, res)
    res_hat = off.predict(Xte)
    fix_yhat = yhat + res_hat
    print("Fixed R2:", r2_score(yte, fix_yhat))
    print("Lift in R2 via offset:", r2_score(yte, fix_yhat) - r2_score(yte, yhat))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=2)
    args=ap.parse_args()
    main(args)
