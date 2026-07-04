"""
Stock Intelligence & Portfolio Risk Dashboard
==============================================
A self-contained Streamlit app that (all with NO external/paid API key):
 
  1. Pulls full-history price data from Yahoo Finance (yfinance)
  2. Builds a monthly seasonality pivot table + classic Support/Resistance pivot points
  3. Computes RSI and MACD
  4. Flags overbought / oversold zones using a mean-reversion (z-score) technique
  5. Pulls fundamental data (P/E, ROE, margins, growth, debt, etc.)
  6. Produces a rule-based "AI Smartness" Buy / Sell / Hold recommendation
     (a transparent, weighted scoring engine - no external AI API required)
  7. Lets a user upload a portfolio (CSV), runs live analysis on it
  8. Computes historical + parametric VaR (downside risk) and the mirror
     upside-VaR (upside potential) for the portfolio
  9. Gives AI-Smartness suggestions on how to rebalance the portfolio
 
Deploy for free:  GitHub repo  +  https://share.streamlit.io
See README.md for step by step instructions.
 
Disclaimer: Educational tool only. Not financial advice.
"""
 
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import yfinance as yf
 
# Check once whether matplotlib is available (pandas Styler.background_gradient needs it).
# If it's ever missing in a given environment, we gracefully fall back to a plain
# (un-colored) dataframe instead of crashing the whole app.
try:
    import matplotlib  # noqa: F401
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
 
 
def safe_dataframe(df_or_styler, use_container_width=True, hide_index=False, **kwargs):
    """
    Render a dataframe/Styler with st.dataframe. If it's a Styler that relies on
    background_gradient (needs matplotlib) and matplotlib isn't installed, or any
    other styling error occurs, fall back to the plain underlying dataframe so the
    app never crashes.
    """
    try:
        st.dataframe(df_or_styler, use_container_width=use_container_width,
                     hide_index=hide_index, **kwargs)
    except Exception:
        plain = getattr(df_or_styler, "data", df_or_styler)
        st.dataframe(plain, use_container_width=use_container_width,
                     hide_index=hide_index, **kwargs)
 
# --------------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Intelligence & Portfolio Risk Dashboard",
    page_icon="📊",
    layout="wide",
)
 
# --------------------------------------------------------------------------------------
# DATA FETCH HELPERS  (cached so we don't hammer Yahoo Finance)
# --------------------------------------------------------------------------------------
 
@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_price_history(ticker: str, period: str = "max") -> pd.DataFrame:
    """Full available price history for a ticker."""
    df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    # yfinance sometimes returns tz-aware dates - strip tz for simplicity
    if pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = df["Date"].dt.tz_localize(None)
    return df
 
 
@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_fundamentals(ticker: str) -> dict:
    """Pull the .info dict (fundamental snapshot) for a ticker."""
    try:
        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}
 
 
@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_last_price(ticker: str) -> float:
    try:
        df = yf.Ticker(ticker).history(period="5d")
        if df.empty:
            return np.nan
        return float(df["Close"].iloc[-1])
    except Exception:
        return np.nan
 
 
# --------------------------------------------------------------------------------------
# TECHNICAL INDICATORS
# --------------------------------------------------------------------------------------
 
def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)
 
 
def calculate_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram
 
 
def mean_reversion_zscore(close: pd.Series, window: int = 20) -> pd.Series:
    rolling_mean = close.rolling(window).mean()
    rolling_std = close.rolling(window).std()
    z = (close - rolling_mean) / rolling_std.replace(0, np.nan)
    return z
 
 
def classic_pivot_points(prev_high, prev_low, prev_close):
    """Standard floor-trader pivot points -> returns dict of P, R1-R3, S1-S3."""
    p = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * p - prev_low
    s1 = 2 * p - prev_high
    r2 = p + (prev_high - prev_low)
    s2 = p - (prev_high - prev_low)
    r3 = prev_high + 2 * (p - prev_low)
    s3 = prev_low - 2 * (prev_high - p)
    return {"Pivot": p, "R1": r1, "R2": r2, "R3": r3, "S1": s1, "S2": s2, "S3": s3}
 
 
# --------------------------------------------------------------------------------------
# "AI SMARTNESS" — TRANSPARENT RULE-BASED SCORING ENGINE (no external API)
# --------------------------------------------------------------------------------------
 
def ai_stock_recommendation(rsi_val, macd_hist_val, macd_hist_prev, zscore_val,
                             price, sma50, sma200, info: dict):
    """
    Weighted scoring engine combining momentum, mean-reversion and fundamentals.
    Returns (recommendation_str, total_score, breakdown_list)
    """
    breakdown = []
    score = 0
 
    # --- Momentum: RSI ---
    if pd.notna(rsi_val):
        if rsi_val < 30:
            score += 2; breakdown.append(("RSI oversold (<30)", +2))
        elif rsi_val > 70:
            score -= 2; breakdown.append(("RSI overbought (>70)", -2))
        else:
            breakdown.append(("RSI neutral", 0))
 
    # --- Momentum: MACD histogram direction ---
    if pd.notna(macd_hist_val) and pd.notna(macd_hist_prev):
        if macd_hist_val > 0 and macd_hist_val > macd_hist_prev:
            score += 1; breakdown.append(("MACD histogram rising & positive", +1))
        elif macd_hist_val < 0 and macd_hist_val < macd_hist_prev:
            score -= 1; breakdown.append(("MACD histogram falling & negative", -1))
        else:
            breakdown.append(("MACD histogram mixed", 0))
 
    # --- Mean reversion z-score ---
    if pd.notna(zscore_val):
        if zscore_val < -1.5:
            score += 1; breakdown.append(("Price far below 20d mean (oversold)", +1))
        elif zscore_val > 1.5:
            score -= 1; breakdown.append(("Price far above 20d mean (overbought)", -1))
        else:
            breakdown.append(("Price near 20d mean", 0))
 
    # --- Trend: price vs 50/200 DMA ---
    if pd.notna(sma50) and pd.notna(sma200):
        if price > sma50 > sma200:
            score += 1; breakdown.append(("Uptrend: Price > 50DMA > 200DMA", +1))
        elif price < sma50 < sma200:
            score -= 1; breakdown.append(("Downtrend: Price < 50DMA < 200DMA", -1))
        else:
            breakdown.append(("Trend mixed/sideways", 0))
 
    # --- Fundamentals ---
    pe = info.get("trailingPE")
    if pe and pe > 0:
        if pe < 25:
            score += 1; breakdown.append((f"Reasonable P/E ({pe:.1f})", +1))
        elif pe > 60:
            score -= 1; breakdown.append((f"Very high P/E ({pe:.1f})", -1))
 
    margins = info.get("profitMargins")
    if margins is not None:
        if margins > 0.10:
            score += 1; breakdown.append((f"Healthy profit margin ({margins*100:.1f}%)", +1))
        elif margins < 0:
            score -= 1; breakdown.append(("Negative profit margin", -1))
 
    earnings_growth = info.get("earningsGrowth")
    if earnings_growth is not None:
        if earnings_growth > 0.05:
            score += 1; breakdown.append((f"Earnings growing ({earnings_growth*100:.1f}%)", +1))
        elif earnings_growth < -0.05:
            score -= 1; breakdown.append((f"Earnings declining ({earnings_growth*100:.1f}%)", -1))
 
    d2e = info.get("debtToEquity")
    if d2e is not None:
        if d2e > 150:
            score -= 1; breakdown.append((f"High debt/equity ({d2e:.0f})", -1))
        elif d2e < 50:
            score += 1; breakdown.append((f"Low debt/equity ({d2e:.0f})", +1))
 
    # --- Map score to label ---
    if score >= 4:
        rec = "STRONG BUY"
    elif score >= 2:
        rec = "BUY"
    elif score <= -4:
        rec = "STRONG SELL"
    elif score <= -2:
        rec = "SELL"
    else:
        rec = "HOLD"
 
    return rec, score, breakdown
 
 
REC_COLOR = {
    "STRONG BUY": "#0f9d58",
    "BUY": "#34a853",
    "HOLD": "#f4b400",
    "SELL": "#e06666",
    "STRONG SELL": "#cc0000",
}
 
 
# --------------------------------------------------------------------------------------
# PORTFOLIO ANALYTICS
# --------------------------------------------------------------------------------------
 
@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_returns_matrix(tickers: tuple, period: str = "2y") -> pd.DataFrame:
    """Daily % returns for each ticker, aligned on date."""
    data = {}
    for t in tickers:
        h = yf.Ticker(t).history(period=period, auto_adjust=True)
        if h is None or h.empty:
            continue
        s = h["Close"].pct_change().dropna()
        s.index = s.index.tz_localize(None) if s.index.tz is not None else s.index
        data[t] = s
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data).dropna(how="all")
 
 
def compute_var(portfolio_returns: pd.Series, portfolio_value: float, confidence: float = 0.95):
    """
    Returns dict with historical + parametric downside VaR and mirror upside-VaR,
    expressed both as % and currency amount, for a 1-day horizon.
    """
    portfolio_returns = portfolio_returns.dropna()
    if portfolio_returns.empty:
        return None
 
    alpha = 1 - confidence
 
    # Historical simulation
    hist_downside_pct = np.percentile(portfolio_returns, alpha * 100)       # e.g. 5th pct -> loss
    hist_upside_pct = np.percentile(portfolio_returns, (1 - alpha) * 100)   # e.g. 95th pct -> gain
 
    # Parametric (variance-covariance), assumes normal returns
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    z = stats.norm.ppf(confidence)
    param_downside_pct = mu - z * sigma
    param_upside_pct = mu + z * sigma
 
    return {
        "confidence": confidence,
        "hist_downside_pct": hist_downside_pct,
        "hist_upside_pct": hist_upside_pct,
        "param_downside_pct": param_downside_pct,
        "param_upside_pct": param_upside_pct,
        "hist_downside_value": hist_downside_pct * portfolio_value,
        "hist_upside_value": hist_upside_pct * portfolio_value,
        "param_downside_value": param_downside_pct * portfolio_value,
        "param_upside_value": param_upside_pct * portfolio_value,
        "daily_vol": sigma,
        "annual_vol": sigma * np.sqrt(252),
    }
 
 
def ai_portfolio_suggestions(port_df: pd.DataFrame, corr: pd.DataFrame, var_info: dict):
    """
    Rule based rebalancing suggestions:
      - flag concentration risk (>25% weight in one name)
      - flag high individual volatility vs portfolio average
      - flag high pairwise correlation (>0.8) as diversification risk
      - flag names with weak technical/fundamental AI score suggesting trim
    """
    notes = []
 
    # Concentration
    over_weight = port_df[port_df["Weight %"] > 25]
    for _, row in over_weight.iterrows():
        notes.append(
            f"⚠️ **{row['Ticker']}** is {row['Weight %']:.1f}% of the portfolio — "
            f"concentration risk. Consider trimming toward 15-20% and reallocating the "
            f"proceeds to your lower-conviction, lower-correlation names."
        )
 
    # Volatility outliers
    if "Ann. Volatility %" in port_df.columns:
        avg_vol = port_df["Ann. Volatility %"].mean()
        vol_outliers = port_df[port_df["Ann. Volatility %"] > avg_vol * 1.5]
        for _, row in vol_outliers.iterrows():
            notes.append(
                f"📈 **{row['Ticker']}** annualized volatility ({row['Ann. Volatility %']:.1f}%) "
                f"is well above the portfolio average ({avg_vol:.1f}%). Sizing it smaller "
                f"would reduce total portfolio swing without necessarily cutting expected return."
            )
 
    # Correlation
    if corr is not None and not corr.empty:
        tickers = corr.columns.tolist()
        high_corr_pairs = []
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                c = corr.iloc[i, j]
                if pd.notna(c) and c > 0.8:
                    high_corr_pairs.append((tickers[i], tickers[j], c))
        for a, b, c in high_corr_pairs[:5]:
            notes.append(
                f"🔗 **{a}** and **{b}** move together very closely (correlation {c:.2f}). "
                f"Holding both adds size, not real diversification — consider consolidating "
                f"into whichever has the stronger AI-Smartness score."
            )
 
    # AI score based trims/adds
    if "AI Score" in port_df.columns:
        weak = port_df[port_df["AI Score"] <= -2]
        strong = port_df[port_df["AI Score"] >= 2]
        for _, row in weak.iterrows():
            notes.append(
                f"🔻 **{row['Ticker']}** currently scores **{row['Recommendation']}** on the "
                f"AI-Smartness engine. If the original investment thesis hasn't changed, this "
                f"is a candidate to trim on strength."
            )
        for _, row in strong.iterrows():
            notes.append(
                f"🔺 **{row['Ticker']}** currently scores **{row['Recommendation']}**. If it's "
                f"underweight relative to conviction, this is a candidate to add to."
            )
 
    # VaR based note
    if var_info:
        notes.append(
            f"📉 At 95% confidence, the portfolio's estimated **1-day downside** is "
            f"**{var_info['hist_downside_pct']*100:.2f}%** "
            f"(≈ {var_info['hist_downside_value']:.2f} in currency terms), while the mirror "
            f"**1-day upside** is **{var_info['hist_upside_pct']*100:.2f}%** "
            f"(≈ {var_info['hist_upside_value']:.2f}). "
            f"If this downside exceeds your risk tolerance, consider trimming the highest-volatility "
            f"names identified above or adding an uncorrelated/defensive asset."
        )
 
    if not notes:
        notes.append("✅ No major concentration, correlation, or volatility red flags detected "
                      "at current thresholds. Portfolio looks reasonably balanced.")
 
    return notes
 
 
# --------------------------------------------------------------------------------------
# UI — HEADER
# --------------------------------------------------------------------------------------
 
st.title("📊 Stock Intelligence & Portfolio Risk Dashboard")
st.caption(
    "Live Yahoo Finance data · Technical + Fundamental analysis · Rule-based **AI Smartness** "
    "recommendations · Portfolio VaR — 100% free, no API key required."
)
 
tab_stock, tab_portfolio = st.tabs(["📈 Single Stock Analysis", "💼 Portfolio Analysis"])
 
# ========================================================================================
# TAB 1 — SINGLE STOCK ANALYSIS
# ========================================================================================
with tab_stock:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        ticker_input = st.text_input(
            "Enter a Yahoo Finance ticker symbol",
            value="AAPL",
            help="Examples: AAPL, MSFT, TSLA (US) · RELIANCE.NS, TCS.NS (India, NSE) · "
                 "VOD.L (London) · 7203.T (Tokyo)",
        )
    with col_b:
        period_choice = st.selectbox(
            "History range",
            ["max", "10y", "5y", "2y", "1y", "6mo"],
            index=0,
        )
 
    if ticker_input:
        ticker = ticker_input.strip().upper()
        with st.spinner(f"Fetching data for {ticker}..."):
            df = fetch_price_history(ticker, period=period_choice)
            info = fetch_fundamentals(ticker)
 
        if df.empty:
            st.error("No data found. Check the ticker symbol (Yahoo Finance format) and try again.")
        else:
            df = df.sort_values("Date").reset_index(drop=True)
            long_name = info.get("longName") or info.get("shortName") or ticker
            st.subheader(f"{long_name} ({ticker})")
 
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            chg = latest["Close"] - prev["Close"]
            chg_pct = (chg / prev["Close"] * 100) if prev["Close"] else 0
 
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Last Close", f"{latest['Close']:.2f}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
            m2.metric("52W High", f"{df['High'].tail(252).max():.2f}")
            m3.metric("52W Low", f"{df['Low'].tail(252).min():.2f}")
            m4.metric("Data Points", f"{len(df):,} days (since {df['Date'].iloc[0].date()})")
 
            # ------------- Price chart -------------
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close", line=dict(color="#1f77b4")))
            df["SMA50"] = df["Close"].rolling(50).mean()
            df["SMA200"] = df["Close"].rolling(200).mean()
            fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA50"], name="SMA 50", line=dict(color="orange", width=1)))
            fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA200"], name="SMA 200", line=dict(color="red", width=1)))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                               legend=dict(orientation="h", y=1.02))
            st.plotly_chart(fig, use_container_width=True)
 
            # ------------------------------------------------------------------
            # PIVOT TABLE (seasonality: avg monthly % return by year) + Pivot Points
            # ------------------------------------------------------------------
            st.markdown("### 🔢 Pivot Table & Support / Resistance")
 
            pc1, pc2 = st.columns([1.3, 1])
 
            with pc1:
                st.markdown("**Seasonality Pivot Table** — average monthly return (%) by year")
                dft = df.copy()
                dft["Year"] = dft["Date"].dt.year
                dft["Month"] = dft["Date"].dt.strftime("%b")
                dft["Ret"] = dft["Close"].pct_change() * 100
                pivot = pd.pivot_table(dft, values="Ret", index="Year", columns="Month", aggfunc="sum")
                month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])
                if _HAS_MATPLOTLIB:
                    safe_dataframe(
                        pivot.style.format("{:.1f}", na_rep="-").background_gradient(
                            cmap="RdYlGn", axis=None, vmin=-15, vmax=15
                        ),
                        use_container_width=True,
                    )
                else:
                    safe_dataframe(pivot.round(1), use_container_width=True)
 
            with pc2:
                st.markdown("**Classic Pivot Points** (based on last completed session)")
                pv_basis = st.radio("Basis", ["Daily", "Weekly", "Monthly"], horizontal=True, key="pivot_basis")
                if pv_basis == "Daily":
                    ph, pl, pcl = df["High"].iloc[-2], df["Low"].iloc[-2], df["Close"].iloc[-2]
                elif pv_basis == "Weekly":
                    wk = df.set_index("Date").resample("W-FRI").agg(
                        {"High": "max", "Low": "min", "Close": "last"}
                    ).dropna()
                    ph, pl, pcl = wk["High"].iloc[-2], wk["Low"].iloc[-2], wk["Close"].iloc[-2]
                else:
                    mo = df.set_index("Date").resample("ME").agg(
                        {"High": "max", "Low": "min", "Close": "last"}
                    ).dropna()
                    ph, pl, pcl = mo["High"].iloc[-2], mo["Low"].iloc[-2], mo["Close"].iloc[-2]
 
                pivots = classic_pivot_points(ph, pl, pcl)
                pv_df = pd.DataFrame(
                    {"Level": ["R3", "R2", "R1", "Pivot", "S1", "S2", "S3"],
                     "Price": [pivots["R3"], pivots["R2"], pivots["R1"], pivots["Pivot"],
                               pivots["S1"], pivots["S2"], pivots["S3"]]}
                )
                safe_dataframe(pv_df.style.format({"Price": "{:.2f}"}), use_container_width=True, hide_index=True)
                st.caption(
                    f"Current price **{latest['Close']:.2f}** is "
                    + ("above the pivot → bullish bias short-term."
                       if latest['Close'] > pivots['Pivot']
                       else "below the pivot → bearish bias short-term.")
                )
 
            # ------------------------------------------------------------------
            # RSI & MACD & Mean Reversion
            # ------------------------------------------------------------------
            st.markdown("### 📉 RSI, MACD & Mean-Reversion (Overbought / Oversold)")
 
            df["RSI"] = calculate_rsi(df["Close"])
            macd_line, signal_line, hist = calculate_macd(df["Close"])
            df["MACD"] = macd_line
            df["MACD_Signal"] = signal_line
            df["MACD_Hist"] = hist
            df["ZScore"] = mean_reversion_zscore(df["Close"], window=20)
 
            rc1, rc2 = st.columns(2)
            with rc1:
                rsi_fig = go.Figure()
                rsi_fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI", line=dict(color="purple")))
                rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
                rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
                rsi_fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title="RSI (14)")
                st.plotly_chart(rsi_fig, use_container_width=True)
 
            with rc2:
                macd_fig = go.Figure()
                macd_fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD", line=dict(color="blue")))
                macd_fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_Signal"], name="Signal", line=dict(color="orange")))
                macd_fig.add_trace(go.Bar(x=df["Date"], y=df["MACD_Hist"], name="Histogram", marker_color="grey"))
                macd_fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title="MACD (12,26,9)")
                st.plotly_chart(macd_fig, use_container_width=True)
 
            z_fig = go.Figure()
            z_fig.add_trace(go.Scatter(x=df["Date"].tail(500), y=df["ZScore"].tail(500),
                                        name="Z-Score (20d)", line=dict(color="teal")))
            z_fig.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="Overbought")
            z_fig.add_hline(y=-1.5, line_dash="dash", line_color="green", annotation_text="Oversold")
            z_fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10),
                                title="Mean-Reversion Z-Score vs 20-day Mean (last 500 sessions)")
            st.plotly_chart(z_fig, use_container_width=True)
 
            rsi_now = df["RSI"].iloc[-1]
            z_now = df["ZScore"].iloc[-1]
            cond_col1, cond_col2, cond_col3 = st.columns(3)
            cond_col1.metric("RSI now", f"{rsi_now:.1f}",
                              "Overbought" if rsi_now > 70 else ("Oversold" if rsi_now < 30 else "Neutral"))
            cond_col2.metric("Mean-Reversion Z-Score", f"{z_now:.2f}",
                              "Overbought" if z_now > 1.5 else ("Oversold" if z_now < -1.5 else "Neutral"))
            cond_col3.metric("MACD Histogram", f"{df['MACD_Hist'].iloc[-1]:.3f}",
                              "Bullish" if df["MACD_Hist"].iloc[-1] > 0 else "Bearish")
 
            # ------------------------------------------------------------------
            # FUNDAMENTALS
            # ------------------------------------------------------------------
            st.markdown("### 🏦 Fundamental Analysis")
            if info:
                f1, f2, f3, f4 = st.columns(4)
                f1.metric("Market Cap", f"{info.get('marketCap', 0):,}" if info.get("marketCap") else "N/A")
                f1.metric("Trailing P/E", f"{info.get('trailingPE'):.2f}" if info.get("trailingPE") else "N/A")
                f2.metric("Forward P/E", f"{info.get('forwardPE'):.2f}" if info.get("forwardPE") else "N/A")
                f2.metric("EPS (TTM)", f"{info.get('trailingEps'):.2f}" if info.get("trailingEps") else "N/A")
                f3.metric("Dividend Yield",
                          f"{info.get('dividendYield')*100:.2f}%" if info.get("dividendYield") else "N/A")
                f3.metric("Beta", f"{info.get('beta'):.2f}" if info.get("beta") else "N/A")
                f4.metric("ROE", f"{info.get('returnOnEquity')*100:.1f}%" if info.get("returnOnEquity") else "N/A")
                f4.metric("Debt/Equity", f"{info.get('debtToEquity'):.0f}" if info.get("debtToEquity") else "N/A")
 
                with st.expander("More fundamentals"):
                    fund_table = {
                        "Sector": info.get("sector"),
                        "Industry": info.get("industry"),
                        "Profit Margin": f"{info.get('profitMargins')*100:.1f}%" if info.get("profitMargins") else "N/A",
                        "Revenue Growth (yoy)": f"{info.get('revenueGrowth')*100:.1f}%" if info.get("revenueGrowth") else "N/A",
                        "Earnings Growth (yoy)": f"{info.get('earningsGrowth')*100:.1f}%" if info.get("earningsGrowth") else "N/A",
                        "Free Cash Flow": info.get("freeCashflow"),
                        "Total Cash": info.get("totalCash"),
                        "Total Debt": info.get("totalDebt"),
                        "52W High": info.get("fiftyTwoWeekHigh"),
                        "52W Low": info.get("fiftyTwoWeekLow"),
                    }
                    st.table(pd.DataFrame(fund_table.items(), columns=["Metric", "Value"]))
            else:
                st.info("Fundamental data not available for this ticker from Yahoo Finance.")
 
            # ------------------------------------------------------------------
            # AI SMARTNESS RECOMMENDATION
            # ------------------------------------------------------------------
            st.markdown("### 🤖 AI Smartness Recommendation")
            rec, score, breakdown = ai_stock_recommendation(
                rsi_val=df["RSI"].iloc[-1],
                macd_hist_val=df["MACD_Hist"].iloc[-1],
                macd_hist_prev=df["MACD_Hist"].iloc[-2],
                zscore_val=df["ZScore"].iloc[-1],
                price=latest["Close"],
                sma50=df["SMA50"].iloc[-1],
                sma200=df["SMA200"].iloc[-1],
                info=info,
            )
            color = REC_COLOR.get(rec, "#999999")
            st.markdown(
                f"<div style='padding:14px;border-radius:10px;background:{color}22;"
                f"border:2px solid {color};text-align:center;'>"
                f"<span style='font-size:26px;font-weight:700;color:{color};'>{rec}</span>"
                f"&nbsp;&nbsp;<span style='font-size:15px;color:#555;'>(composite score: {score})</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.caption("Rule-based weighted engine over momentum, mean-reversion and fundamentals — "
                       "runs entirely locally, no external AI API used.")
            bdf = pd.DataFrame(breakdown, columns=["Factor", "Score Contribution"])
            st.dataframe(bdf, use_container_width=True, hide_index=True)
            st.caption("⚠️ Educational tool only — not financial advice. Always do your own research.")
 
 
# ========================================================================================
# TAB 2 — PORTFOLIO ANALYSIS
# ========================================================================================
with tab_portfolio:
    st.markdown("#### Upload your portfolio")
    st.caption(
        "CSV with columns: **Ticker, Quantity, Buy_Price** (Buy_Price optional, used only "
        "to show unrealized P&L). Tickers must be in Yahoo Finance format."
    )
 
    sample_csv = pd.DataFrame(
        {"Ticker": ["AAPL", "MSFT", "RELIANCE.NS"], "Quantity": [10, 5, 20], "Buy_Price": [180.0, 300.0, 2500.0]}
    )
    st.download_button(
        "⬇️ Download sample CSV template",
        sample_csv.to_csv(index=False).encode("utf-8"),
        file_name="sample_portfolio.csv",
        mime="text/csv",
    )
 
    uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"])
 
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            raw = None
 
        if raw is not None:
            raw.columns = [c.strip() for c in raw.columns]
            required = {"Ticker", "Quantity"}
            if not required.issubset(set(raw.columns)):
                st.error(f"CSV must contain at least these columns: {sorted(required)}")
            else:
                raw["Ticker"] = raw["Ticker"].astype(str).str.strip().str.upper()
                if "Buy_Price" not in raw.columns:
                    raw["Buy_Price"] = np.nan
 
                tickers = tuple(raw["Ticker"].unique().tolist())
 
                with st.spinner("Fetching live prices & history for your portfolio..."):
                    live_prices = {t: fetch_last_price(t) for t in tickers}
                    returns_matrix = fetch_returns_matrix(tickers, period="2y")
                    fundamentals_map = {t: fetch_fundamentals(t) for t in tickers}
                    hist_map = {t: fetch_price_history(t, period="2y") for t in tickers}
 
                raw["Live Price"] = raw["Ticker"].map(live_prices)
                raw["Market Value"] = raw["Live Price"] * raw["Quantity"]
                total_value = raw["Market Value"].sum()
                raw["Weight %"] = (raw["Market Value"] / total_value * 100) if total_value else 0
 
                raw["Unrealized P&L"] = np.where(
                    raw["Buy_Price"].notna(),
                    (raw["Live Price"] - raw["Buy_Price"]) * raw["Quantity"],
                    np.nan,
                )
                raw["Unrealized P&L %"] = np.where(
                    raw["Buy_Price"].notna() & (raw["Buy_Price"] != 0),
                    (raw["Live Price"] - raw["Buy_Price"]) / raw["Buy_Price"] * 100,
                    np.nan,
                )
 
                # Per-holding volatility + AI score
                vol_list, score_list, rec_list = [], [], []
                for t in raw["Ticker"]:
                    h = hist_map.get(t, pd.DataFrame())
                    if h is not None and not h.empty and len(h) > 30:
                        h = h.sort_values("Date")
                        close = h["Close"]
                        rsi_s = calculate_rsi(close)
                        macd_l, macd_s, macd_h = calculate_macd(close)
                        z_s = mean_reversion_zscore(close, 20)
                        sma50 = close.rolling(50).mean().iloc[-1]
                        sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
                        daily_ret = close.pct_change().dropna()
                        ann_vol = daily_ret.std() * np.sqrt(252) * 100
                        rec_t, score_t, _ = ai_stock_recommendation(
                            rsi_s.iloc[-1], macd_h.iloc[-1], macd_h.iloc[-2], z_s.iloc[-1],
                            close.iloc[-1], sma50, sma200, fundamentals_map.get(t, {}),
                        )
                    else:
                        ann_vol, score_t, rec_t = np.nan, 0, "HOLD"
                    vol_list.append(ann_vol)
                    score_list.append(score_t)
                    rec_list.append(rec_t)
 
                raw["Ann. Volatility %"] = vol_list
                raw["AI Score"] = score_list
                raw["Recommendation"] = rec_list
 
                st.markdown("### 📋 Live Portfolio Analysis")
                st.metric("Total Portfolio Value", f"{total_value:,.2f}")
 
                display_cols = ["Ticker", "Quantity", "Live Price", "Market Value", "Weight %",
                                 "Unrealized P&L", "Unrealized P&L %", "Ann. Volatility %",
                                 "AI Score", "Recommendation"]
                safe_dataframe(
                    raw[display_cols].style.format({
                        "Live Price": "{:.2f}", "Market Value": "{:.2f}", "Weight %": "{:.1f}",
                        "Unrealized P&L": "{:.2f}", "Unrealized P&L %": "{:.1f}",
                        "Ann. Volatility %": "{:.1f}",
                    }, na_rep="-"),
                    use_container_width=True, hide_index=True,
                )
 
                pie = px.pie(raw, values="Market Value", names="Ticker", title="Portfolio Allocation")
                st.plotly_chart(pie, use_container_width=True)
 
                # ------------------------------------------------------------------
                # VaR
                # ------------------------------------------------------------------
                st.markdown("### 📉 Portfolio Value-at-Risk (VaR)")
                if not returns_matrix.empty:
                    weights = raw.set_index("Ticker")["Weight %"] / 100
                    weights = weights.reindex(returns_matrix.columns).fillna(0)
                    portfolio_returns = (returns_matrix * weights).sum(axis=1)
 
                    confidence = st.slider("Confidence level", 0.90, 0.99, 0.95, 0.01)
                    var_info = compute_var(portfolio_returns, total_value, confidence)
 
                    if var_info:
                        v1, v2, v3, v4 = st.columns(4)
                        v1.metric("1-Day Downside VaR (Historical)",
                                  f"{var_info['hist_downside_pct']*100:.2f}%",
                                  f"{var_info['hist_downside_value']:.2f}")
                        v2.metric("1-Day Upside VaR (Historical)",
                                  f"{var_info['hist_upside_pct']*100:.2f}%",
                                  f"{var_info['hist_upside_value']:.2f}")
                        v3.metric("1-Day Downside VaR (Parametric)",
                                  f"{var_info['param_downside_pct']*100:.2f}%",
                                  f"{var_info['param_downside_value']:.2f}")
                        v4.metric("1-Day Upside VaR (Parametric)",
                                  f"{var_info['param_upside_pct']*100:.2f}%",
                                  f"{var_info['param_upside_value']:.2f}")
                        st.caption(
                            f"Annualized portfolio volatility ≈ **{var_info['annual_vol']*100:.1f}%**. "
                            f"VaR estimates the size of a 1-in-{int(1/(1-confidence))} day move; scale by "
                            f"√t for longer horizons (e.g. multiply the % by √10 for a 10-day estimate)."
                        )
 
                        ret_fig = px.histogram(portfolio_returns * 100, nbins=60,
                                                title="Distribution of Portfolio Daily Returns (%)")
                        ret_fig.add_vline(x=var_info["hist_downside_pct"] * 100, line_color="red",
                                          annotation_text="Downside VaR")
                        ret_fig.add_vline(x=var_info["hist_upside_pct"] * 100, line_color="green",
                                          annotation_text="Upside VaR")
                        st.plotly_chart(ret_fig, use_container_width=True)
 
                        # ------------------------------------------------------------------
                        # AI Smartness — portfolio suggestions
                        # ------------------------------------------------------------------
                        st.markdown("### 🤖 AI Smartness — Suggested Portfolio Changes")
                        corr = returns_matrix.corr()
                        notes = ai_portfolio_suggestions(raw, corr, var_info)
                        for n in notes:
                            st.markdown(n)
 
                        with st.expander("Correlation matrix between holdings"):
                            if _HAS_MATPLOTLIB:
                                safe_dataframe(
                                    corr.style.format("{:.2f}").background_gradient(cmap="RdYlGn_r"),
                                    use_container_width=True,
                                )
                            else:
                                safe_dataframe(corr.round(2), use_container_width=True)
                else:
                    st.warning("Not enough historical return data to compute VaR for these tickers.")
 
    else:
        st.info("Upload a CSV to run live portfolio analysis, or download the sample template above to see the required format.")
 
st.markdown("---")
st.caption(
    "Data: Yahoo Finance via `yfinance` (free, unofficial). All indicators, VaR and recommendations are "
    "computed locally with open-source math — no paid or external AI API key is used anywhere in this app. "
    "This tool is for educational purposes only and is not investment advice."
)
