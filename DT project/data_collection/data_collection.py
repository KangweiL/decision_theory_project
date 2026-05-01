import os
import sys
import numpy as np
import pandas as pd
import requests

# ------------------------------------------------------------
# 1) Configuration
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_fred_api_key():
    """Load the FRED API key from the environment or a local project file."""
    env_key = os.getenv("FRED_API_KEY")
    if env_key:
        return env_key.strip()

    env_path = os.path.join(SCRIPT_DIR, ".env")
    if os.path.isfile(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                name, value = line.split("=", 1)
                if name.strip() == "FRED_API_KEY":
                    return value.strip().strip('"').strip("'")

    key_path = os.path.join(SCRIPT_DIR, "fred_api_key.txt")
    if os.path.isfile(key_path):
        with open(key_path, "r", encoding="utf-8") as f:
            key = f.read().strip()
            if key:
                return key

    return None


FRED_API_KEY = load_fred_api_key()
if not FRED_API_KEY:
    sys.exit(
        "FRED_API_KEY is not set. Put it in one of these locations:\n"
        "  1) environment variable FRED_API_KEY\n"
        "  2) a .env file next to this script containing FRED_API_KEY=your_api_key_here\n"
        "  3) a fred_api_key.txt file next to this script containing just the key\n"
        "Then rerun the script."
    )

START = "2016-01-01"
END = "2026-04-29"

SERIES = {
    "sp500": "SP500",              # S&P 500 index
    "dgs10": "DGS10",              # 10Y Treasury yield
    "dgs2": "DGS2",                # 2Y Treasury yield
    "hy_oas": "BAMLH0A0HYM2",      # High-yield OAS
    "ig_oas": "BAMLC0A0CM",        # Investment-grade corporate OAS
    "wti": "DCOILWTICO",           # WTI crude oil spot price
    "vix": "VIXCLS",               # VIX
    "dollar": "DTWEXBGS",          # Broad nominal USD index (daily)
    # alternative monthly dollar series:
    # "dollar": "TWEXBGSMTH",
}

# ------------------------------------------------------------
# 2) FRED downloader
# ------------------------------------------------------------
def fetch_fred_series(series_id: str,
                      start: str,
                      end: str,
                      api_key: str,
                      frequency: str = None,
                      aggregation_method: str = None) -> pd.Series:
    """
    Download one FRED series using the official observations endpoint.
    Returns a pandas Series indexed by date.
    """
    if not api_key:
        raise ValueError("Set FRED_API_KEY in your environment first.")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }

    # Optional FRED-side aggregation
    if frequency is not None:
        params["frequency"] = frequency
    if aggregation_method is not None:
        params["aggregation_method"] = aggregation_method

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["observations"]

    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError(f"No data returned for {series_id}")

    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.set_index("date")["value"].sort_index()
    s.name = series_id
    return s

# ------------------------------------------------------------
# 3) Download raw daily data
# ------------------------------------------------------------
raw = {}
for name, series_id in SERIES.items():
    try:
        raw[name] = fetch_fred_series(
            series_id=series_id,
            start=START,
            end=END,
            api_key=FRED_API_KEY
        )
    except Exception as exc:
        raise RuntimeError(f"Failed downloading FRED series '{series_id}' for '{name}': {exc}") from exc

raw_daily = pd.concat(raw.values(), axis=1)
raw_daily.columns = list(raw.keys())

# ------------------------------------------------------------
# 4) Basic cleaning
# ------------------------------------------------------------
# Replace obvious missing markers already handled by to_numeric(..., errors="coerce")
# Forward fill only where it makes economic sense after resampling, not here.
raw_daily = raw_daily.sort_index()

# ------------------------------------------------------------
# 5) Convert to monthly
# ------------------------------------------------------------
# Prices / indexes: use month-end
price_cols = ["sp500", "wti", "dollar"]

# Yields / spreads / volatility: use monthly average
avg_cols = ["dgs10", "dgs2", "hy_oas", "ig_oas", "vix"]

monthly_prices = raw_daily[price_cols].resample("M").last()
monthly_avgs = raw_daily[avg_cols].resample("M").mean()

monthly = pd.concat([monthly_prices, monthly_avgs], axis=1)

# Optional: forward fill limited gaps after monthly conversion
monthly = monthly.ffill(limit=1)

# ------------------------------------------------------------
# 6) Build regime features
# ------------------------------------------------------------
features = pd.DataFrame(index=monthly.index)

# Monthly log returns
features["equity_ret"] = np.log(monthly["sp500"]).diff()
features["oil_ret"] = np.log(monthly["wti"]).diff()
features["dollar_ret"] = np.log(monthly["dollar"]).diff()

# Rates / spreads / vol
features["vix_level"] = monthly["vix"]
features["hy_oas"] = monthly["hy_oas"]
features["ig_oas"] = monthly["ig_oas"]
features["term_spread"] = monthly["dgs10"] - monthly["dgs2"]

# Bond signal: change in 10Y yield (negative often corresponds to bond rally)
features["d10_change"] = monthly["dgs10"].diff()

# Simple trend feature derived from equity data
# Example: trailing 12-month log return
features["trend_12m"] = np.log(monthly["sp500"]).diff(12)

# ------------------------------------------------------------
# 7) Final cleanup
# ------------------------------------------------------------
features = features.drop(columns=["hy_oas", "ig_oas"])
features = features.replace([np.inf, -np.inf], np.nan).dropna()

# ------------------------------------------------------------
# 8) Optional robust scaling for clustering / regime models
# ------------------------------------------------------------
def robust_zscore(df: pd.DataFrame) -> pd.DataFrame:
    med = df.median()
    iqr = df.quantile(0.75) - df.quantile(0.25)
    iqr = iqr.replace(0, np.nan)
    return (df - med) / iqr

features_scaled = robust_zscore(features).dropna()

# ------------------------------------------------------------
# 9) Save outputs
# ------------------------------------------------------------
raw_daily.to_csv("raw_daily_market_regime_data.csv")
monthly_raw = monthly.drop(columns=["hy_oas", "ig_oas"])
monthly_raw.to_csv("monthly_raw_market_regime_data.csv")
features.to_csv("monthly_regime_features.csv")
features_scaled.to_csv("monthly_regime_features_scaled.csv")

print("Raw daily shape:", raw_daily.shape)
print("Monthly raw shape:", monthly_raw.shape)
print("Feature shape:", features.shape)
print(features.tail())

print(raw_daily.loc["2018-07-01":"2018-07-31", ["hy_oas", "ig_oas"]])