# 📊 Stock Intelligence & Portfolio Risk Dashboard

A free, no-API-key Streamlit app that pulls data straight from Yahoo Finance and gives you:

- Full price history (all available timeframe) with SMA 50/200 overlay
- A seasonality **pivot table** (avg monthly return by year) + classic **pivot-point Support/Resistance** (R1-R3 / S1-S3, daily/weekly/monthly basis)
- **RSI** and **MACD**
- **Overbought / Oversold** detection using a **mean-reversion z-score** technique (price distance from its 20-day mean)
- **Fundamental analysis** (P/E, EPS, ROE, margins, growth, debt/equity, dividend yield, etc.)
- A transparent, rule-based **"AI Smartness"** Buy / Sell / Hold engine — a weighted scoring system, no external AI API used
- A **Portfolio tab**: upload a CSV of your holdings and get live valuation, per-holding volatility, and AI scores
- **Portfolio VaR** (Value at Risk): historical + parametric downside VaR, and the mirror **upside VaR**, at a confidence level you choose
- AI-Smartness **rebalancing suggestions** (concentration risk, correlation risk, volatility outliers, weak/strong scorers)

> ⚠️ Educational tool only. Not financial advice. Yahoo Finance data via the unofficial `yfinance` library can occasionally be delayed, rate-limited, or incomplete.

---

## 1. Run locally

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

---

## 2. Deploy for free — GitHub + share.streamlit.io

### Step A — Push the code to GitHub
1. Create a new **public** (or private, if you have a paid Streamlit plan) GitHub repository, e.g. `stock-ai-dashboard`.
2. Add these three files to the repo root:
   - `app.py`
   - `requirements.txt`
   - (optional) `sample_portfolio.csv`, `README.md`
3. Commit and push:
   ```bash
   git init
   git add app.py requirements.txt README.md sample_portfolio.csv
   git commit -m "Initial commit: Stock AI dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

### Step B — Deploy on Streamlit Community Cloud
1. Go to **https://share.streamlit.io** and sign in with your GitHub account.
2. Click **"New app"**.
3. Choose:
   - **Repository:** `<your-username>/<your-repo>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy**. Streamlit Cloud will read `requirements.txt` automatically and install everything.
5. In a minute or two you'll get a public URL like:
   `https://<your-app-name>.streamlit.app`

### Step C — Redeploying after changes
Every time you `git push` to the connected branch, Streamlit Community Cloud automatically redeploys the app — no extra steps needed.

---

## 3. Portfolio CSV format

| Column      | Required | Notes                                                            |
|-------------|----------|-------------------------------------------------------------------|
| `Ticker`    | Yes      | Yahoo Finance symbol, e.g. `AAPL`, `RELIANCE.NS`, `TCS.NS`, `VOD.L` |
| `Quantity`  | Yes      | Number of shares held                                            |
| `Buy_Price` | No       | Your average buy price, used only to display unrealized P&L      |

A ready-to-use template is included (`sample_portfolio.csv`) and is also downloadable directly from the app's **Portfolio Analysis** tab.

---

## 4. Notes on the indicators used

- **RSI (14)** — Wilder's smoothing via an exponential moving average of gains/losses.
- **MACD (12, 26, 9)** — standard EMA-based MACD line, signal line, and histogram.
- **Mean-reversion z-score** — `(Close − 20-day rolling mean) / 20-day rolling std`. `z > 1.5` is flagged overbought, `z < -1.5` oversold.
- **Pivot points** — classic floor-trader formula using the previous completed period's High/Low/Close: `P = (H+L+C)/3`, with `R1/R2/R3` and `S1/S2/S3` derived from `P`.
- **VaR** — both a non-parametric **historical simulation** (5th/95th percentile of realized daily portfolio returns) and a **parametric (variance-covariance)** estimate assuming normally distributed returns, at a user-selectable confidence level (default 95%).
- **AI Smartness engine** — a fully transparent, weighted scoring model that combines RSI, MACD momentum, the mean-reversion z-score, moving-average trend, and fundamental health (P/E, margins, earnings growth, debt/equity) into a single composite score, mapped to STRONG BUY → STRONG SELL. It runs 100% locally — there's no external LLM/AI API call, no key required, and every point in the score is shown to you in a breakdown table so it's auditable, not a black box.

---

## 5. Limitations / disclaimers

- Yahoo Finance (via `yfinance`) is an **unofficial** data source; it can rate-limit or return incomplete data occasionally — just retry.
- Fundamentals (`.info`) coverage varies by exchange/ticker; some fields may show `N/A`.
- The "AI Smartness" score is a rule-based heuristic, not a trained machine-learning model and not investment advice.
- VaR assumes future volatility resembles the recent historical window (2 years by default) and does not capture tail/black-swan risk fully.
