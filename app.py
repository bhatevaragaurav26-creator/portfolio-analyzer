# app.py — Portfolio Analyzer Pro (2025-ready, fixed)
# ---------------------------------------------------
# Features:
# - Currency conversion to chosen base (detect per-asset currency; Yahoo FX pairs)
# - Two portfolios (A & B) with side-by-side metrics and charts
# - Historical VaR / CVaR (configurable confidence)
# - Factor/beta decomposition vs selected benchmarks (OLS)
# - Streamlit deprecation handled: uses width="stretch"/"content" (no use_container_width)
#
# Requirements:
#   pip install -r requirements.txt
#   (streamlit, yfinance, pandas, numpy, plotly, reportlab)

import io
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px

# ---------------- Page ----------------
st.set_page_config(page_title="Portfolio Analyzer Pro", layout="wide")
st.title("📊 Portfolio Analyzer Pro")
st.caption("Currency conversion, A/B compare, VaR/CVaR, and factor betas — updated for Streamlit’s width= API.")

# ---------------- Helpers ----------------
def to_dt(d):
    if isinstance(d, datetime):
        return d
    if isinstance(d, date):
        return datetime.combine(d, datetime.min.time())
    return datetime.today()

def _np_isfinite(x):
    try:
        return np.isfinite(x)
    except Exception:
        return False

def cagr_from_prices(pr: pd.Series) -> float:
    s = pr.dropna()
    if s.shape[0] < 2 or s.iloc[0] <= 0:
        return np.nan
    years = (s.index[-1] - s.index[0]).days / 365.25
    if years <= 0:
        return np.nan
    return (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1

def ann_vol(ret: pd.Series, per_year=252) -> float:
    r = ret.dropna()
    if r.empty:
        return np.nan
    return r.std(ddof=0) * np.sqrt(per_year)

def sharpe(ret: pd.Series, rf=0.0, per_year=252) -> float:
    r = ret.dropna()
    if r.empty:
        return np.nan
    ex = r - (rf / per_year)
    vol = ex.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return (ex.mean() * per_year) / vol

def max_drawdown_from_returns(ret: pd.Series) -> float:
    r = ret.dropna()
    if r.empty:
        return np.nan
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1
    return dd.min()

def hist_var_cvar(ret: pd.Series, cl=0.95):
    """Historical VaR/CVaR of daily returns; positive value = % loss."""
    r = ret.dropna()
    if r.empty:
        return (np.nan, np.nan)
    q = np.quantile(r, 1 - cl)  # e.g., 5th pct for 95% CL
    var = -q
    cvar = -r[r <= q].mean() if (r <= q).any() else np.nan
    return (var, cvar)

def safe_download_prices(tickers, start, end, interval="1d"):
    """Adj Close in native currency per asset."""
    if not tickers:
        return pd.DataFrame()
    try:
        data = yf.download(
            tickers=tickers,
            start=start.date(),
            end=(end + timedelta(days=1)).date(),  # include end
            interval=interval,
            auto_adjust=True,
            progress=False,
            group_by="ticker"
        )
        # Normalize to simple DataFrame [Date x tickers]
        if isinstance(data.columns, pd.MultiIndex):
            cols = []
            lvl1 = {l for _, l in data.columns}
            preferred = "Adj Close" if "Adj Close" in lvl1 else ("Close" if "Close" in lvl1 else None)
            if preferred:
                for t in tickers:
                    if (t, preferred) in data.columns:
                        cols.append(data[(t, preferred)].rename(t))
            prices = pd.concat(cols, axis=1) if cols else pd.DataFrame(index=data.index)
        else:
            # Single series fallback
            single = data.rename(columns={"Adj Close": tickers[0], "Close": tickers[0]})
            prices = single[[tickers[0]]] if tickers[0] in single.columns else pd.DataFrame(index=single.index)
        prices = prices.sort_index().ffill().dropna(how="all")
        return prices
    except Exception as e:
        st.error(f"Price download error: {e}")
        return pd.DataFrame()

def detect_currency(ticker: str) -> str:
    # Try yfinance fast_info
    try:
        info = yf.Ticker(ticker).fast_info
        cur = info.get("currency")
        if cur:
            return cur.upper()
    except Exception:
        pass
    # Heuristics by suffix
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return "INR"
    if ticker.endswith(".L"):
        return "GBP"
    if ticker.endswith(".TO"):
        return "CAD"
    if ticker.endswith(".HK"):
        return "HKD"
    if ticker.endswith(".AX"):
        return "AUD"
    if ticker.endswith(".TW"):
        return "TWD"
    if ticker.endswith(".T"):
        return "JPY"
    return "USD"

def fetch_fx_pair_series(curr_from: str, curr_to: str, start: datetime, end: datetime, interval="1d") -> pd.Series:
    """Return FX series as 'units of TO per 1 FROM'. If same currency → ones."""
    cf = curr_from.upper()
    ct = curr_to.upper()
    if cf == ct:
        idx = pd.date_range(start=start.date(), end=end.date(), freq="D")
        return pd.Series(1.0, index=idx, name=f"{cf}{ct}=X")
    pair = f"{cf}{ct}=X"  # e.g., USDINR=X
    try:
        df = yf.download(pair, start=start.date(), end=(end + timedelta(days=1)).date(),
                         interval=interval, progress=False, auto_adjust=False)
        s = (df["Adj Close"] if "Adj Close" in df.columns else df["Close"]).rename(pair).ffill()
        return s
    except Exception:
        return pd.Series(dtype=float, name=pair)

def convert_prices_to_base(prices: pd.DataFrame, tickers: list, base_ccy: str, start: datetime, end: datetime, interval="1d"):
    """Convert each asset column to base currency using detected currency and Yahoo FX pairs."""
    if prices.empty:
        return prices, {t: "?" for t in tickers}, {}

    asset_ccy = {t: detect_currency(t) for t in tickers}
    fx_cache = {}
    conv = pd.DataFrame(index=prices.index)
    for t in tickers:
        ccy = asset_ccy.get(t, "USD").upper()
        key = (ccy, base_ccy.upper())
        if key not in fx_cache:
            fx_cache[key] = fetch_fx_pair_series(ccy, base_ccy, start, end, interval=interval)
        fx = fx_cache[key].reindex(conv.index).ffill()
        conv[t] = prices[t] * fx
    return conv, asset_ccy, {f"{k[0]}->{k[1]}": v for k, v in fx_cache.items()}

def tracking_error(a_ret: pd.Series, b_ret: pd.Series, per_year=252):
    df = pd.concat([a_ret, b_ret], axis=1).dropna()
    if df.shape[0] < 3:
        return np.nan
    diff = df.iloc[:, 0] - df.iloc[:, 1]
    return diff.std(ddof=0) * np.sqrt(per_year)

def beta_alpha(asset: pd.Series, bench: pd.Series, per_year=252):
    df = pd.concat([asset, bench], axis=1).dropna()
    if df.shape[0] < 3 or df.iloc[:, 1].std(ddof=0) == 0:
        return (np.nan, np.nan, np.nan)
    x = df.iloc[:, 1].values
    y = df.iloc[:, 0].values
    X = np.column_stack([np.ones_like(x), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha_per, beta1 = b[0], b[1]
    yhat = X @ b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    alpha_ann = (1 + alpha_per) ** per_year - 1
    return (beta1, alpha_ann, r2)

def multi_factor_ols(y: pd.Series, X_df: pd.DataFrame, per_year=252):
    """OLS with intercept; returns dict(alpha_ann, betas{col:beta}, r2)."""
    df = pd.concat([y.rename("y"), X_df], axis=1).dropna()
    if df.shape[0] < (X_df.shape[1] + 2):
        return {"alpha_ann": np.nan, "betas": {c: np.nan for c in X_df.columns}, "r2": np.nan}
    Y = df["y"].values
    X = df[X_df.columns].values
    X_ = np.column_stack([np.ones(X.shape[0]), X])
    b = np.linalg.lstsq(X_, Y, rcond=None)[0]
    alpha_per = b[0]
    betas = {col: b[i+1] for i, col in enumerate(X_df.columns)}
    yhat = X_ @ b
    ss_res = np.sum((Y - yhat) ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    alpha_ann = (1 + alpha_per) ** per_year - 1
    return {"alpha_ann": alpha_ann, "betas": betas, "r2": r2}

def fmt_pct(x, digits=2):
    return "—" if x is None or not _np_isfinite(x) else f"{x:.{digits}%}"

def fmt_num(x, digits=2):
    return "—" if x is None or not _np_isfinite(x) else f"{x:.{digits}f}"

# ---- FIXED: summarize_portfolio signature matches calls (no unused params) ----
def summarize_portfolio(spec, prices_base, rf):
    """Compute returns/metrics for a single portfolio dict with keys: label,tickers,weights."""
    tickers = spec["tickers"]
    w = np.array(spec["weights"], dtype=float)
    if w.sum() == 0 or not np.isfinite(w).all():
        return (pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), {})
    w = w / w.sum()

    pr = prices_base[tickers].dropna(how="all")
    if pr.empty:
        return (pr, pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), {})

    rets = pr.pct_change().dropna(how="all")
    port_ret = (rets * w).sum(axis=1).rename(f"{spec['label']}_ret")

    rows = []
    for i, t in enumerate(tickers):
        s = pr[t].dropna()
        r = rets[t].dropna()
        rows.append({
            "Ticker": t,
            "Weight": w[i],
            "CAGR": cagr_from_prices(s),
            "Annual Vol": ann_vol(r),
            "Sharpe": sharpe(r, rf=rf),
            "Max Drawdown": max_drawdown_from_returns(r),
        })
    mdf = pd.DataFrame(rows)
    pm = {
        "CAGR": cagr_from_prices((pr @ w)),
        "Annual Vol": ann_vol(port_ret),
        "Sharpe": sharpe(port_ret, rf=rf),
        "Max Drawdown": max_drawdown_from_returns(port_ret),
    }
    return pr, rets, port_ret, mdf, pm

def make_pdf_report(buffer: io.BytesIO, title: str, summary_blocks: list, tables: dict):
    """Write a simple PDF; returns True on success (fallback handled by caller)."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
        for head, items in summary_blocks:
            story.append(Paragraph(f"<b>{head}</b>", styles["Heading2"]))
            for k, v in items.items():
                story.append(Paragraph(f"{k}: {v}", styles["Normal"]))
            story.append(Spacer(1, 8))
        for name, df in tables.items():
            story.append(Paragraph(f"<b>{name}</b>", styles["Heading3"]))
            tdata = [df.columns.tolist()] + df.astype(str).values.tolist()
            tbl = Table(tdata, hAlign="LEFT")
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
                ("ALIGN", (1,1), (-1,-1), "RIGHT"),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 8))
        doc.build(story)
        return True
    except Exception:
        return False

# ---------------- Sidebar: Global Config ----------------
with st.sidebar:
    st.header("⚙️ Configuration")
    base_currency = st.selectbox("Base currency", ["USD", "INR", "EUR", "GBP"], index=1)
    interval = st.selectbox("Price interval", ["1d", "1wk", "1mo"], index=0)
    rf_rate = st.number_input("Risk-free rate (annual, %)", value=0.0, step=0.25) / 100.0
    var_cl = st.slider("VaR/CVaR Confidence", min_value=0.80, max_value=0.999, value=0.95, step=0.01)
    d1, d2 = st.columns(2)
    with d1:
        start_dt = st.date_input("Start date", value=(date.today() - timedelta(days=365*3)))
    with d2:
        end_dt = st.date_input("End date", value=date.today())

# ---------------- Sidebar: Portfolio A ----------------
with st.sidebar:
    st.markdown("---")
    st.subheader("🅰️ Portfolio A")
    a_tickers_text = st.text_input("A: Tickers", value="AAPL, MSFT")
    a_tickers = [t.strip() for t in a_tickers_text.split(",") if t.strip()]
    a_weights_text = st.text_input(
        "A: Weights",
        value=", ".join([str(round(1/len(a_tickers), 4)) for _ in a_tickers]) if a_tickers else ""
    )
    try:
        a_weights = [float(x.strip()) for x in a_weights_text.split(",")] if a_weights_text else []
    except Exception:
        a_weights = []

# ---------------- Sidebar: Portfolio B ----------------
with st.sidebar:
    st.subheader("🅱️ Portfolio B (optional)")
    b_tickers_text = st.text_input("B: Tickers", value="^NSEI, ^BSESN")
    b_tickers = [t.strip() for t in b_tickers_text.split(",") if t.strip()]
    b_weights_text = st.text_input(
        "B: Weights",
        value=", ".join([str(round(1/len(b_tickers), 4)) for _ in b_tickers]) if b_tickers else ""
    )
    try:
        b_weights = [float(x.strip()) for x in b_weights_text.split(",")] if b_weights_text else []
    except Exception:
        b_weights = []

# ---------------- Sidebar: Benchmarks ----------------
with st.sidebar:
    st.markdown("---")
    st.subheader("🧭 Benchmarks / Factors")
    bench_opts = {
        "^NSEI": "NIFTY 50",
        "^BSESN": "SENSEX",
        "^GSPC": "S&P 500",
        "^NDX": "Nasdaq 100",
        "^FTSE": "FTSE 100",
        "^STOXX50E": "Euro Stoxx 50",
        "^N225": "Nikkei 225",
        "^HSI": "Hang Seng",
    }
    selected_bench = st.multiselect(
        "Select up to 4 benchmarks",
        options=list(bench_opts.keys()),
        default=["^NSEI", "^BSESN"],
        max_selections=4,
        format_func=lambda x: f"{x} ({bench_opts.get(x,'')})"
    )
    run_btn = st.button("Run Analysis")

# ---------------- Main Logic ----------------
if run_btn:
    start = to_dt(start_dt)
    end = to_dt(end_dt)
    if start >= end:
        st.error("Start date must be before end date.")
        st.stop()

    # Validate A
    if not a_tickers or not a_weights or len(a_tickers) != len(a_weights):
        st.error("Portfolio A: please provide tickers and matching weights.")
        st.stop()
    use_B = bool(b_tickers) and bool(b_weights) and len(b_tickers) == len(b_weights)

    # Download prices for all assets
    all_tickers = list(dict.fromkeys(a_tickers + (b_tickers if use_B else [])))
    raw_prices = safe_download_prices(all_tickers, start, end, interval=interval)
    if raw_prices.empty:
        st.error("No price data returned. Check tickers/date range.")
        st.stop()

    # Currency conversion to base
    conv_prices, asset_ccy_map, fx_map = convert_prices_to_base(raw_prices, all_tickers, base_currency, start, end, interval=interval)

    with st.expander("💱 Currency details", expanded=False):
        st.write(f"**Base currency:** {base_currency}")
        st.dataframe(pd.DataFrame({"Ticker": list(asset_ccy_map.keys()),
                                   "Detected Currency": list(asset_ccy_map.values())}), width="content")

    # Benchmarks (converted to base)
    if selected_bench:
        bench_prices_native = safe_download_prices(selected_bench, start, end, interval=interval)
        if not bench_prices_native.empty:
            bench_conv = []
            for b in selected_bench:
                bc = detect_currency(b)
                fx_b = fetch_fx_pair_series(bc, base_currency, start, end, interval=interval)
                bench_conv.append((bench_prices_native[b] * fx_b.reindex(bench_prices_native.index).ffill()).rename(b))
            bench_prices = pd.concat(bench_conv, axis=1).dropna(how="all")
            bench_rets = bench_prices.pct_change().dropna(how="all")
        else:
            bench_prices = pd.DataFrame()
            bench_rets = pd.DataFrame()
    else:
        bench_prices = pd.DataFrame()
        bench_rets = pd.DataFrame()

    # Summaries
    a_spec = {"label": "A", "tickers": a_tickers, "weights": a_weights}
    A_prices, A_rets, A_port_ret, A_metrics_df, A_pm = summarize_portfolio(a_spec, conv_prices, rf_rate)

    if use_B:
        b_spec = {"label": "B", "tickers": b_tickers, "weights": b_weights}
        B_prices, B_rets, B_port_ret, B_metrics_df, B_pm = summarize_portfolio(b_spec, conv_prices, rf_rate)
    else:
        B_prices = pd.DataFrame(); B_rets = pd.DataFrame()
        B_port_ret = pd.Series(dtype=float, name="B_ret")
        B_metrics_df = pd.DataFrame(); B_pm = {}

    # -------- Top Summary --------
    st.subheader(f"📌 Portfolio Summaries (base: {base_currency})")
    cols = st.columns(8)
    cols[0].metric("A CAGR", fmt_pct(A_pm.get("CAGR")))
    cols[1].metric("A Vol", fmt_pct(A_pm.get("Annual Vol")))
    cols[2].metric("A Sharpe", fmt_num(A_pm.get("Sharpe")))
    cols[3].metric("A MaxDD", fmt_pct(A_pm.get("Max Drawdown")))
    if use_B:
        cols[4].metric("B CAGR", fmt_pct(B_pm.get("CAGR")))
        cols[5].metric("B Vol", fmt_pct(B_pm.get("Annual Vol")))
        cols[6].metric("B Sharpe", fmt_num(B_pm.get("Sharpe")))
        cols[7].metric("B MaxDD", fmt_pct(B_pm.get("Max Drawdown")))

    st.subheader("🧮 Asset Metrics")
    left, right = st.columns(2)
    with left:
        st.markdown("**Portfolio A**")
        if not A_metrics_df.empty:
            dfA = A_metrics_df.copy()
            dfA["Weight"] = dfA["Weight"].map(fmt_pct)
            for col in ["CAGR", "Annual Vol", "Max Drawdown"]:
                dfA[col] = dfA[col].map(fmt_pct)
            dfA["Sharpe"] = dfA["Sharpe"].map(fmt_num)
            st.dataframe(dfA, width="stretch")
        else:
            st.info("No data for Portfolio A.", icon="ℹ️")
    with right:
        st.markdown("**Portfolio B**")
        if not B_metrics_df.empty:
            dfB = B_metrics_df.copy()
            dfB["Weight"] = dfB["Weight"].map(fmt_pct)
            for col in ["CAGR", "Annual Vol", "Max Drawdown"]:
                dfB[col] = dfB[col].map(fmt_pct)
            dfB["Sharpe"] = dfB["Sharpe"].map(fmt_num)
            st.dataframe(dfB, width="stretch")
        else:
            st.info("No Portfolio B configured.", icon="ℹ️")

    # -------- Charts --------
    st.subheader("📈 Charts")

    # Converted prices
    fig_price = px.line(conv_prices[all_tickers], x=conv_prices.index, y=all_tickers, title=f"Converted Prices in {base_currency}")
    fig_price.update_layout(legend_title_text="")
    st.plotly_chart(fig_price, width="stretch")

    # Cumulative returns (A, B, benchmarks)
    cum_df = pd.DataFrame(index=A_rets.index)
    if not A_port_ret.empty:
        cum_df["A"] = (1 + A_port_ret).cumprod()
    if use_B and not B_port_ret.empty:
        cum_df["B"] = (1 + B_port_ret.reindex(cum_df.index).dropna()).cumprod()
    if not bench_rets.empty:
        for b in bench_rets.columns:
            cum_df[b] = (1 + bench_rets[b].reindex(cum_df.index)).cumprod()
    cum_df = cum_df.dropna(how="all")
    if not cum_df.empty:
        fig_cum = px.line(cum_df, x=cum_df.index, y=cum_df.columns, title="Cumulative Returns")
        fig_cum.update_layout(legend_title_text="")
        st.plotly_chart(fig_cum, width="stretch")

    # Risk/Return bubble — assets in A & B
    rr_rows = []
    if not A_prices.empty:
        wA = np.array(a_weights, dtype=float); wA = wA/ wA.sum()
        for i, t in enumerate(a_tickers):
            r = A_rets[t].dropna()
            rr_rows.append({"Portfolio": "A", "Ticker": t, "Annual Vol": ann_vol(r), "CAGR": cagr_from_prices(A_prices[t]), "Weight": wA[i]})
    if use_B and not B_prices.empty:
        wB = np.array(b_weights, dtype=float); wB = wB/ wB.sum()
        for i, t in enumerate(b_tickers):
            r = B_rets[t].dropna()
            rr_rows.append({"Portfolio": "B", "Ticker": t, "Annual Vol": ann_vol(r), "CAGR": cagr_from_prices(B_prices[t]), "Weight": wB[i]})
    rr_df = pd.DataFrame(rr_rows)
    if not rr_df.empty:
        fig_bub = px.scatter(rr_df, x="Annual Vol", y="CAGR", size="Weight", color="Portfolio", text="Ticker",
                             title="Risk / Return by Asset (bubble=size weight)")
        fig_bub.update_traces(textposition="top center")
        fig_bub.update_layout(xaxis_tickformat=".1%", yaxis_tickformat=".1%", legend_title_text="")
        st.plotly_chart(fig_bub, width="stretch")

    # -------- Risk: VaR / CVaR --------
    st.subheader("⚠️ Risk: Historical VaR / CVaR")
    var_rows = []
    if not A_port_ret.empty:
        A_var, A_cvar = hist_var_cvar(A_port_ret, cl=var_cl)
        var_rows.append(["Portfolio A", fmt_pct(A_var), fmt_pct(A_cvar)])
    if use_B and not B_port_ret.empty:
        B_var, B_cvar = hist_var_cvar(B_port_ret, cl=var_cl)
        var_rows.append(["Portfolio B", fmt_pct(B_var), fmt_pct(B_cvar)])
    if var_rows:
        st.dataframe(pd.DataFrame(var_rows, columns=["Portfolio", f"VaR ({int(var_cl*100)}%)", f"CVaR ({int(var_cl*100)}%)"]), width="content")

    # -------- Factor / Beta Decomposition --------
    st.subheader("🧮 Factor / Beta Decomposition vs Benchmarks")
    if not bench_rets.empty:
        cL, cR = st.columns(2)
        with cL:
            st.markdown("**Portfolio A vs each benchmark (single-factor)**")
            rows = []
            for b in bench_rets.columns:
                b1, a1, r2 = beta_alpha(A_port_ret, bench_rets[b])
                rows.append([f"{b} ({bench_opts.get(b,'')})", fmt_num(b1), fmt_pct(a1), fmt_num(r2)])
            st.dataframe(pd.DataFrame(rows, columns=["Benchmark", "Beta", "Alpha (annual)", "R²"]), width="content")
        with cR:
            st.markdown("**Portfolio B vs each benchmark (single-factor)**")
            if use_B and not B_port_ret.empty:
                rowsB = []
                for b in bench_rets.columns:
                    b1, a1, r2 = beta_alpha(B_port_ret, bench_rets[b])
                    rowsB.append([f"{b} ({bench_opts.get(b,'')})", fmt_num(b1), fmt_pct(a1), fmt_num(r2)])
                st.dataframe(pd.DataFrame(rowsB, columns=["Benchmark", "Beta", "Alpha (annual)", "R²"]), width="content")
            else:
                st.info("No Portfolio B configured.", icon="ℹ️")

        # Multi-factor (all benchmarks together)
        st.markdown("**Multi-factor regression (all selected benchmarks together)**")
        X = bench_rets.copy()
        A_mf = multi_factor_ols(A_port_ret, X)
        mfA = [["Alpha (annual)", fmt_pct(A_mf["alpha_ann"])], ["R²", fmt_num(A_mf["r2"])]] + \
              [[f"β vs {b} ({bench_opts.get(b,'')})", fmt_num(v)] for b, v in A_mf["betas"].items()]
        st.dataframe(pd.DataFrame(mfA, columns=["Metric", "Value"]), width="content")

        if use_B and not B_port_ret.empty:
            B_mf = multi_factor_ols(B_port_ret, X)
            mfB = [["Alpha (annual)", fmt_pct(B_mf["alpha_ann"])], ["R²", fmt_num(B_mf["r2"])]] + \
                  [[f"β vs {b} ({bench_opts.get(b,'')})", fmt_num(v)] for b, v in B_mf["betas"].items()]
            st.dataframe(pd.DataFrame(mfB, columns=["Metric", "Value"]), width="content")
    else:
        st.info("Select at least one benchmark to see factor/beta decomposition.", icon="ℹ️")

    # -------- Downloads --------
    st.subheader("⬇️ Downloads")
    export = pd.concat({
        "Prices (base)": conv_prices,
        "A returns": A_rets if not A_rets.empty else pd.DataFrame(index=conv_prices.index),
        "B returns": B_rets if not B_rets.empty else pd.DataFrame(index=conv_prices.index),
        "Bench returns": bench_rets if not bench_rets.empty else pd.DataFrame(index=conv_prices.index)
    }, axis=1)
    csv_buf = io.StringIO()
    export.to_csv(csv_buf)
    st.download_button("Download CSV (prices & returns)", data=csv_buf.getvalue(),
                       file_name="portfolio_pro_data.csv", mime="text/csv")

    pdf_buf = io.BytesIO()
    A_sum = {k: (fmt_pct(v) if k in ["CAGR","Annual Vol","Max Drawdown"] else (fmt_num(v) if k=="Sharpe" else v))
             for k, v in A_pm.items()}
    blocks = [("Portfolio A Summary", A_sum)]
    if use_B and B_pm:
        B_sum = {k: (fmt_pct(v) if k in ["CAGR","Annual Vol","Max Drawdown"] else (fmt_num(v) if k=="Sharpe" else v))
                 for k, v in B_pm.items()}
        blocks.append(("Portfolio B Summary", B_sum))
    tables = {}
    if not A_metrics_df.empty:
        dfA_out = A_metrics_df.copy()
        dfA_out["Weight"] = dfA_out["Weight"].map(lambda x: f"{x:.2%}")
        for c in ["CAGR","Annual Vol","Max Drawdown"]:
            dfA_out[c] = dfA_out[c].map(lambda x: f"{x:.2%}" if _np_isfinite(x) else "—")
        dfA_out["Sharpe"] = dfA_out["Sharpe"].map(lambda x: f"{x:.2f}" if _np_isfinite(x) else "—")
        tables["Asset Metrics - A"] = dfA_out
    if use_B and not B_metrics_df.empty:
        dfB_out = B_metrics_df.copy()
        dfB_out["Weight"] = dfB_out["Weight"].map(lambda x: f"{x:.2%}")
        for c in ["CAGR","Annual Vol","Max Drawdown"]:
            dfB_out[c] = dfB_out[c].map(lambda x: f"{x:.2%}" if _np_isfinite(x) else "—")
        dfB_out["Sharpe"] = dfB_out["Sharpe"].map(lambda x: f"{x:.2f}" if _np_isfinite(x) else "—")
        tables["Asset Metrics - B"] = dfB_out

    if make_pdf_report(pdf_buf, "Portfolio Analyzer Pro Report", blocks, tables):
        st.download_button("Download PDF Report", data=pdf_buf.getvalue(),
                           file_name="portfolio_pro_report.pdf", mime="application/pdf")
    else:
        html = io.StringIO()
        html.write("<h1>Portfolio Analyzer Pro Report</h1>")
        for head, items in blocks:
            html.write(f"<h2>{head}</h2><ul>")
            for k, v in items.items():
                html.write(f"<li><b>{k}</b>: {v}</li>")
            html.write("</ul>")
        for name, df in tables.items():
            html.write(f"<h2>{name}</h2>")
            html.write(df.to_html(index=False))
        st.download_button("Download HTML Report", data=html.getvalue().encode("utf-8"),
                           file_name="portfolio_pro_report.html", mime="text/html")

else:
    st.info("Set base currency, portfolios, dates, and click **Run Analysis**.", icon="🛠️")
