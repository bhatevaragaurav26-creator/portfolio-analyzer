import numpy as np, pandas as pd, yfinance as yf
from datetime import datetime, timedelta

def _dl(tickers, start, end, interval="1d"):
    df = yf.download(tickers=tickers, start=start, end=(end+timedelta(days=1)),
                     interval=interval, auto_adjust=True, progress=False, group_by="ticker")
    if isinstance(df.columns, pd.MultiIndex):
        parts = []
        pick = "Adj Close" if "Adj Close" in {c for _, c in df.columns} else "Close"
        for t in tickers:
            if (t, pick) in df.columns: parts.append(df[(t, pick)].rename(t))
        out = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    else:
        t = tickers[0]
        out = df.rename(columns={"Adj Close": t, "Close": t})[[t]]
    return out.sort_index().ffill()

def cagr(pr: pd.Series):
    s = pr.dropna()
    if len(s)<2 or s.iloc[0]<=0: return np.nan
    yrs = (s.index[-1]-s.index[0]).days/365.25
    return (s.iloc[-1]/s.iloc[0])**(1/yrs)-1 if yrs>0 else np.nan

def ann_vol(r: pd.Series, n=252): 
    r=r.dropna();  return r.std(ddof=0)*np.sqrt(n) if not r.empty else np.nan

def sharpe(r: pd.Series, rf=0.0, n=252):
    r=r.dropna(); 
    if r.empty: return np.nan
    ex=r-(rf/n); s=ex.std(ddof=0)
    return (ex.mean()*n)/s if s not in [0,np.nan] else np.nan

def mdd(r: pd.Series):
    r=r.dropna(); 
    if r.empty: return np.nan
    cum=(1+r).cumprod(); peak=cum.cummax()
    return (cum/peak-1).min()

def var_cvar(r: pd.Series, cl=0.95):
    r=r.dropna(); 
    if r.empty: return (np.nan,np.nan)
    q=np.quantile(r,1-cl); return (-q, -r[r<=q].mean() if (r<=q).any() else np.nan)

def analyze(tickers, weights, start, end, rf=0.0, cl=0.95, interval="1d"):
    prices=_dl(tickers, start, end, interval)
    rets=prices.pct_change().dropna(how="all")
    w=np.array(weights, dtype=float); w/=w.sum()
    port_ret=(rets*w).sum(axis=1)

    assets=[]
    for i,t in enumerate(tickers):
        s=prices[t]; r=rets[t]
        assets.append({
            "ticker": t,
            "weight": float(w[i]),
            "cagr": float(cagr(s)),
            "vol": float(ann_vol(r)),
            "sharpe": float(sharpe(r, rf)),
            "mdd": float(mdd(r)),
        })

    p = prices @ w
    V,CV = var_cvar(port_ret, cl)
    summary = {
      "cagr": float(cagr(p)),
      "vol": float(ann_vol(port_ret)),
      "sharpe": float(sharpe(port_ret, rf)),
      "mdd": float(mdd(port_ret)),
      "var": float(V), "cvar": float(CV)
    }
    return prices, rets, port_ret, assets, summary
