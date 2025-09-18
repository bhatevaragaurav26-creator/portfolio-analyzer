# app.py  — Portfolio Analyzer (robust FX + validation + MC)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date

# ---------- Page ----------
st.set_page_config(page_title="Portfolio Analyzer", layout="wide")
st.title("Portfolio Analyzer")

# ---------- Core helpers ----------
def to_dt(d):
    if isinstance(d, datetime): return d
    if isinstance(d, date): return datetime.combine(d, datetime.min.time())
    return datetime.today()

def cagr(start_val, end_val, start_dt, end_dt):
    years = (end_dt - start_dt).days / 365.25
    return (end_val/start_val)**(1/years) - 1 if start_val > 0 and years > 0 else np.nan

def annualize_vol(returns: pd.Series):
    return returns.std(ddof=1) * np.sqrt(252) if not returns.empty else np.nan

def compute_beta(rp: pd.Series, rb: pd.Series):
    df = pd.concat([rp, rb], axis=1).dropna()
    if len(df) < 30: return np.nan
    cov = np.cov(df.iloc[:,0], df.iloc[:,1])[0,1]
    varb = np.var(df.iloc[:,1])
    return cov/varb if varb > 0 else np.nan

def fetch_adj_close(tickers, start, end):
    """Fetch adjusted close; returns columns=tickers, index=date."""
    tickers = [t.strip().upper() for t in tickers if str(t).strip()]
    if not tickers: return pd.DataFrame()
    data = yf.download(tickers, start=start, end=end, auto_adjust=True,
                       progress=False, group_by="ticker", threads=True)
    prices = None
    if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
        lvl1 = data.columns.get_level_values(1)
        field = "Adj Close" if "Adj Close" in lvl1 else ("Close" if "Close" in lvl1 else None)
        prices = data.xs(field, axis=1, level=1) if field is not None else data.xs(lvl1.unique()[0], axis=1, level=1)
    elif isinstance(data, pd.DataFrame):
        if "Adj Close" in data.columns:
            s = data["Adj Close"]; prices = s.to_frame() if isinstance(s, pd.Series) else s
            if len(tickers) == 1: prices.columns = [tickers[0]]
        elif "Close" in data.columns:
            s = data["Close"]; prices = s.to_frame() if isinstance(s, pd.Series) else s
            if len(tickers) == 1: prices.columns = [tickers[0]]
        else:
            prices = data.copy()
    else:
        prices = pd.DataFrame()
    if prices is None or prices.empty: return pd.DataFrame()
    if isinstance(prices, pd.Series): prices = prices.to_frame()
    return prices.sort_index().ffill().dropna(how="all")

def portfolio_series(prices: pd.DataFrame, amounts_map: dict) -> pd.Series:
    """Allocate by invested amounts at start -> infer shares -> sum to series."""
    first = prices.ffill().bfill().iloc[0]
    shares = {}
    for t, amt in amounts_map.items():
        if t in first and float(first[t]) > 0:
            shares[t] = float(amt) / float(first[t])
    if not shares: return pd.Series(dtype=float)
    return (prices[list(shares.keys())] * pd.Series(shares)).sum(axis=1)

# ---------- Validation with suggestions ----------
COMMON_SUFFIXES = [".L",".NS",".NSE",".BO",".TO",".V",".AX",".HK",".PA",".F",".SW",".SA",".MI",".SG",".TW",".KS",".KQ"]

def has_recent_data(tkr):
    try:
        df = yf.download(tkr, period="5d", interval="1d", progress=False, auto_adjust=True)
        return isinstance(df, pd.DataFrame) and df.shape[0] >= 1
    except Exception:
        return False

def suggest_variants(ticker):
    out = []
    for suf in COMMON_SUFFIXES:
        cand = f"{ticker}{suf}"
        if has_recent_data(cand):
            out.append(cand)
        if len(out) >= 5: break
    return out

def validate_and_suggest(tickers):
    valid, invalid, suggestions = [], [], {}
    for t in tickers:
        if has_recent_data(t): valid.append(t)
        else:
            invalid.append(t)
            s = suggest_variants(t)
            if s: suggestions[t] = s
    return valid, invalid, suggestions

# ---------- Cached metadata ----------
@st.cache_data(ttl=60*60*12, show_spinner=False)
def cached_sector(ticker):
    try:
        info = yf.Ticker(ticker).info
        return (info.get("sector") or "Unknown")
    except Exception:
        return "Unknown"

def get_sectors(tickers):
    return {t: cached_sector(t) for t in tickers}

@st.cache_data(ttl=60*60*12, show_spinner=False)
def cached_currency(ticker):
    try:
        info = yf.Ticker(ticker).info
        cur = info.get("currency")
        return str(cur).upper() if cur else "USD"
    except Exception:
        return "USD"

# ---------- Robust FX conversion ----------
def _fx_series_yahoo(pair, start, end):
    df = yf.download(pair, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df, pd.DataFrame) and "Adj Close" in df.columns:
        return df["Adj Close"].rename(pair).ffill()
    if isinstance(df, pd.DataFrame) and "Close" in df.columns:
        return df["Close"].rename(pair).ffill()
    return pd.Series(dtype=float)

def fetch_fx_pair_robust(base_ccy, tgt_ccy, start, end):
    """Return a Series converting 1 unit of base_ccy -> tgt_ccy.
       Try BASETGT=X, else TGTPASE=X and invert."""
    base = (base_ccy or "USD").upper().strip()
    tgt  = (tgt_ccy  or "USD").upper().strip()
    if base == tgt:
        return pd.Series(dtype=float).rename(f"{base}{tgt}=X")
    direct = f"{base}{tgt}=X"
    fx = _fx_series_yahoo(direct, start, end)
    if fx is not None and not fx.empty:
        return fx
    inv = _fx_series_yahoo(f"{tgt}{base}=X", start, end)
    if inv is not None and not inv.empty:
        inv = inv.replace(0, np.nan).dropna()
        if not inv.empty:
            return (1.0 / inv).rename(direct)
    return pd.Series(dtype=float).rename(direct)

def convert_prices_to_target_robust(prices_df, start, end, target_ccy, currency_map, *, on_fail="error"):
    """Convert each ticker column from its native currency -> target_ccy.
       on_fail='error'  -> raise ValueError listing failed tickers
       on_fail='skip'   -> drop failed tickers"""
    tgt = (target_ccy or "USD").upper().strip()
    if prices_df.empty:
        return prices_df
    converted = prices_df.copy()
    failed = []
    for t in list(prices_df.columns):
        base = (currency_map.get(t) or "USD").upper().strip()
        if base == tgt:
            continue
        fx = fetch_fx_pair_robust(base, tgt, start, end)
        if fx is None or fx.empty:
            failed.append((t, base)); continue
        aligned = pd.concat([prices_df[t].rename("P"), fx.rename("FX")], axis=1).dropna()
        if aligned.empty:
            failed.append((t, base)); continue
        converted.loc[aligned.index, t] = aligned["P"] * aligned["FX"]
    if failed:
        msg = ", ".join([f"{t}({b})" for t, b in failed])
        if on_fail == "skip":
            to_drop = [t for t, _ in failed]
            converted.drop(columns=to_drop, inplace=True, errors="ignore")
        else:
            raise ValueError(f"FX conversion missing for: {msg}")
    return converted.dropna(axis=1, how="all")

# ---------- Monte Carlo ----------
def simulate_gbm_from_series(port_series, years_ahead=5, n_paths=5000, seed=42):
    s = port_series.dropna()
    if s.shape[0] < 30:
        return None, "Not enough history (need at least ~30 daily observations)."
    daily = s.pct_change().dropna()
    mu_daily = daily.mean(); sigma_daily = daily.std(ddof=1)
    mu_ann = (1 + mu_daily) ** 252 - 1
    sigma_ann = sigma_daily * np.sqrt(252)
    if not np.isfinite(mu_ann) or not np.isfinite(sigma_ann) or sigma_ann <= 0:
        return None, "Could not estimate drift/volatility."
    current = float(s.iloc[-1])
    steps = int(252 * years_ahead); dt = 1.0/252.0
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(steps, n_paths))
    drift = (mu_ann - 0.5 * sigma_ann**2) * dt
    diff = sigma_ann * np.sqrt(dt) * Z
    log_cum = np.cumsum(drift + diff, axis=0)
    end_values = current * np.exp(log_cum[-1, :])
    p5, p50, p95 = np.percentile(end_values, [5,50,95]).tolist()
    def to_cagr(v_end, v_start, yrs):
        return (v_end/v_start)**(1/yrs) - 1 if v_start>0 and yrs>0 else np.nan
    out = {
        "current_value": current,
        "p5_value": p5, "p50_value": p50, "p95_value": p95,
        "cagr_p5": to_cagr(p5,current,years_ahead),
        "cagr_p50": to_cagr(p50,current,years_ahead),
        "cagr_p95": to_cagr(p95,current,years_ahead),
        "end_values": end_values
    }
    return out, None

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    benchmark = st.text_input("Benchmark", value="^GSPC")
    risk_free = st.number_input("Risk-free rate (annual)", value=0.02, step=0.005, format="%.4f")
    target_currency = st.text_input("Display currency (USD, GBP, EUR, INR, etc.)", value="USD")
st.caption("Tip: Enter global tickers (AAPL, MSFT, NVDA, TSLA, RIO.L, RELIANCE.NS).")

# ---------- Inputs ----------
c1, c2 = st.columns(2)
with c1:
    tickers_input = st.text_input("Tickers (comma-separated)", value="AAPL, MSFT, NVDA")
    start_date = st.date_input("Start date", value=datetime(2022,1,1))
with c2:
    amounts_input = st.text_input("Invested amounts (same order)", value="4000, 3500, 2500")
    end_date = st.date_input("End date", value=datetime.today())

# ---------- Analyze ----------
if st.button("Analyze"):
    import traceback
    try:
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        amounts = [float(x.strip()) for x in amounts_input.split(",") if x.strip()]
        if len(tickers) != len(amounts):
            st.error("Tickers and amounts count must match."); st.stop()

        # Validate (with suggestions)
        valid, invalid, suggestions = validate_and_suggest(tickers)
        if invalid:
            msg = "Invalid or no recent data: " + ", ".join(invalid)
            if suggestions:
                tips = "; ".join([f"{k} -> {', '.join(v)}" for k,v in suggestions.items()])
                msg += " | Suggestions: " + tips
            st.warning(msg)
        if not valid: st.error("No valid tickers to analyze."); st.stop()
        tickers = valid

        # Prices
        prices = fetch_adj_close(tickers, start_date, end_date)
        if prices.empty: st.error("No price data returned."); st.stop()

        available = [t for t in tickers if t in prices.columns]
        if not available: st.error("No valid tickers with data."); st.stop()
        prices = prices[available]
        amounts_aligned = [a for (t,a) in zip(tickers, amounts) if t in available]
        amounts_map = dict(zip(available, amounts_aligned))

        # Per-ticker currency conversion to display currency (robust)
        currency_map = {t: cached_currency(t) for t in prices.columns}
        prices = convert_prices_to_target_robust(prices, start_date, end_date, target_currency, currency_map, on_fail="error")
        if prices.empty: st.error("Currency conversion failed for all tickers."); st.stop()

        # Portfolio series & metrics
        port = portfolio_series(prices, amounts_map)
        if port.empty: st.error("Could not build portfolio series."); st.stop()

        total_invested = float(sum(amounts_aligned))
        current_value = float(port.iloc[-1])
        abs_return = (current_value - total_invested)/total_invested if total_invested>0 else np.nan
        cagr_val = cagr(total_invested, current_value, to_dt(start_date), to_dt(end_date))
        daily = port.pct_change().dropna()
        vol_ann = annualize_vol(daily)

        # Benchmark & beta
        bench = fetch_adj_close([benchmark], start_date, end_date)
        if not bench.empty:
            bser = bench[benchmark] if benchmark in bench.columns else bench.iloc[:,0]
            beta_val = compute_beta(daily, bser.pct_change().dropna())
        else:
            beta_val = np.nan
            st.warning("Benchmark data unavailable; beta not computed.")

        # Sharpe
        if not daily.empty and vol_ann and not np.isnan(vol_ann):
            ann_ret = (1 + daily).prod() ** (252/len(daily)) - 1
            sharpe = (ann_ret - risk_free) / vol_ann
        else:
            sharpe = np.nan

        # KPIs
        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Invested", f"{target_currency} {total_invested:,.2f}")
        k2.metric("Current", f"{target_currency} {current_value:,.2f}")
        k3.metric("Abs. Return", f"{abs_return*100:,.2f}%")
        k4.metric("CAGR", f"{cagr_val*100:,.2f}%")
        k5.metric("Volatility (ann.)", f"{(vol_ann*100) if vol_ann==vol_ann else 0:.2f}%")
        k6,k7 = st.columns(2)
        k6.metric("Sharpe", f"{sharpe:.3f}" if sharpe==sharpe else "—")
        k7.metric("Beta vs Benchmark", f"{beta_val:.3f}" if beta_val==beta_val else "—")

        st.subheader("Portfolio value (display currency)")
        st.line_chart(port, height=320)

        # Sector allocation by invested weights
        st.subheader("Sector allocation")
        weights = {t: (amt/total_invested if total_invested>0 else 0.0) for t, amt in amounts_map.items()}
        sectors_map = get_sectors(list(weights.keys()))
        sec = {}
        for t, w in weights.items():
            sct = sectors_map.get(t, "Unknown") or "Unknown"
            sec[sct] = sec.get(sct, 0.0) + w
        sec_df = pd.DataFrame([{"sector":k,"weight":v} for k,v in sec.items()]).sort_values("weight", ascending=False)
        if not sec_df.empty:
            st.bar_chart(sec_df.set_index("sector"))
            st.dataframe(sec_df.assign(weight_pct=(sec_df["weight"]*100).round(2)))
        else:
            st.info("No sector data found.")

        # Downloads
        def _kpis_df():
            return pd.DataFrame([{
                "currency": target_currency,
                "invested": round(total_invested,2),
                "current": round(current_value,2),
                "abs_return_pct": round(abs_return*100,2) if pd.notna(abs_return) else None,
                "cagr_pct": round(cagr_val*100,2) if pd.notna(cagr_val) else None,
                "vol_ann_pct": round(vol_ann*100,2) if pd.notna(vol_ann) else None,
                "sharpe": round(sharpe,3) if pd.notna(sharpe) else None,
                "beta": round(beta_val,3) if pd.notna(beta_val) else None
            }])

        def _csv_bytes(df): return df.to_csv(index=False).encode("utf-8")
        def _xlsx_bytes(df, sheet="KPIs"):
            import io
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
                df.to_excel(w, index=False, sheet_name=sheet)
            buf.seek(0); return buf.getvalue()

        kdf = _kpis_df()
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("Download KPIs (CSV)", data=_csv_bytes(kdf), file_name="kpis_single.csv", mime="text/csv")
        with d2:
            st.download_button("Download KPIs (Excel)", data=_xlsx_bytes(kdf), file_name="kpis_single.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except Exception as ex:
        import traceback
        st.error(f"Analysis failed: {ex}")
        st.code(traceback.format_exc())

# ---------- Monte Carlo ----------
st.header("Monte Carlo simulation")
colA,colB,colC = st.columns(3)
with colA:
    sim_years = st.number_input("Years ahead", 1, 30, 5, 1)
with colB:
    sim_paths = st.number_input("Number of paths", 500, 50000, 5000, 500)
with colC:
    sim_seed = st.number_input("Random seed", 0, 999999, 42, 1)

if st.button("Run simulation"):
    try:
        tickers = [t.strip().upper() for t in (tickers_input or "").split(",") if t.strip()]
        amounts = [float(x.strip()) for x in (amounts_input or "").split(",") if x.strip()]
        if len(tickers) != len(amounts) or len(tickers) == 0:
            st.error("Please fill tickers and amounts above, then click Analyze first."); st.stop()

        prices_sim = fetch_adj_close(tickers, start_date, end_date)
        if prices_sim.empty: st.error("No price data for simulation."); st.stop()
        available = [t for t in tickers if t in prices_sim.columns]
        if not available: st.error("No valid tickers for simulation."); st.stop()
        prices_sim = prices_sim[available]
        amounts = [a for (t,a) in zip(tickers,amounts) if t in available]
        amounts_map_sim = dict(zip(available, amounts))

        currency_map_sim = {t: cached_currency(t) for t in prices_sim.columns}
        prices_sim = convert_prices_to_target_robust(prices_sim, start_date, end_date, target_currency, currency_map_sim, on_fail="error")
        if prices_sim.empty: st.error("Currency conversion failed for simulation."); st.stop()

        port_sim = portfolio_series(prices_sim, amounts_map_sim)
        out, err = simulate_gbm_from_series(port_sim, years_ahead=int(sim_years), n_paths=int(sim_paths), seed=int(sim_seed))
        if err: st.error(err); st.stop()

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Current", f"{target_currency} {out['current_value']:,.2f}")
        m2.metric("P5", f"{target_currency} {out['p5_value']:,.2f}")
        m3.metric("P50", f"{target_currency} {out['p50_value']:,.2f}")
        m4.metric("P95", f"{target_currency} {out['p95_value']:,.2f}")

        n1,n2,n3 = st.columns(3)
        n1.metric("CAGR P5",  f"{out['cagr_p5']*100:.2f}%")
        n2.metric("CAGR P50", f"{out['cagr_p50']*100:.2f}%")
        n3.metric("CAGR P95", f"{out['cagr_p95']*100:.2f}%")

        # Histogram
        ev = out["end_values"]
        counts, bins = np.histogram(ev, bins=40)
        mids = (bins[:-1] + bins[1:]) / 2.0
        hist_df = pd.DataFrame({"end_value": mids, "count": counts}).set_index("end_value")
        st.subheader("Distribution of simulated end values")
        st.bar_chart(hist_df)

        # Downloads
        sim_summary = pd.DataFrame([{
            "current_value": round(out["current_value"],2),
            "p5_value": round(out["p5_value"],2),
            "p50_value": round(out["p50_value"],2),
            "p95_value": round(out["p95_value"],2),
            "cagr_p5_pct": round(out["cagr_p5"]*100,2),
            "cagr_p50_pct": round(out["cagr_p50"]*100,2),
            "cagr_p95_pct": round(out["cagr_p95"]*100,2),
        }])
        def _csv_bytes(df): return df.to_csv(index=False).encode("utf-8")
        def _xlsx_bytes(df, sheet="Simulation"):
            import io
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
                df.to_excel(w, index=False, sheet_name=sheet)
            buf.seek(0); return buf.getvalue()
        s1,s2 = st.columns(2)
        with s1:
            st.download_button("Download simulation (CSV)", data=_csv_bytes(sim_summary),
                               file_name="simulation_summary.csv", mime="text/csv")
        with s2:
            st.download_button("Download simulation (Excel)", data=_xlsx_bytes(sim_summary),
                               file_name="simulation_summary.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as ex:
        import traceback
        st.error(f"Simulation failed: {ex}")
        st.code(traceback.format_exc())
