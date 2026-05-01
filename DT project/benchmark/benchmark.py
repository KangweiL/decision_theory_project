import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

ROOT = "/Users/likangwei/Desktop/DT project"

FINAL_FP = os.path.join(ROOT, "Final_result_2", "final_regime_probabilities_em_v2.csv")
RAW_MONTHLY_FP = os.path.join(ROOT, "data_collection", "monthly_raw_market_regime_data.csv")
OUT_DIR = os.path.join(ROOT, "benchmark")

# =========================================================
# 1. Load regime results
# =========================================================
df = pd.read_csv(FINAL_FP)
df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp("M")
df = df.sort_values("date")

# =========================================================
# 2. Load SP500
# =========================================================
raw = pd.read_csv(RAW_MONTHLY_FP)
raw["date"] = pd.to_datetime(raw["date"]).dt.to_period("M").dt.to_timestamp("M")

def find_sp500_column(columns):
    for c in columns:
        if "sp" in c.lower() or "close" in c.lower():
            return c
    return None

sp_col = find_sp500_column(raw.columns)
raw = raw[["date", sp_col]].rename(columns={sp_col: "sp500"})

df = pd.merge(df, raw, on="date", how="inner")

# =========================================================
# 3. Compute drawdown
# =========================================================
df["running_max"] = df["sp500"].cummax()
df["drawdown"] = (df["sp500"] - df["running_max"]) / df["running_max"]

# define stress benchmark
threshold = -0.15
df["drawdown_stress"] = (df["drawdown"] < threshold).astype(int)

# =========================================================
# 4. Model prediction
# =========================================================
df["pred_stress"] = (df["assigned_regime"].str.lower() == "stress").astype(int)
df["score"] = df["prob_stress"]

# =========================================================
# 5. Metrics
# =========================================================
y_true = df["drawdown_stress"]
y_pred = df["pred_stress"]
y_score = df["score"]

metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred),
    "recall": recall_score(y_true, y_pred),
    "f1": f1_score(y_true, y_pred),
    "roc_auc": roc_auc_score(y_true, y_score)
}

print(pd.DataFrame([metrics]).round(4))

# =========================================================
# 6. Visualization
# =========================================================
fig, ax = plt.subplots(figsize=(13,5))

# stress regime shading
for i in range(len(df)):
    if df["pred_stress"].iloc[i]:
        ax.axvspan(df["date"].iloc[i], df["date"].iloc[i] + pd.offsets.MonthEnd(1),
                   color="green", alpha=0.4)

# drawdown shading
for i in range(len(df)):
    if df["drawdown_stress"].iloc[i]:
        ax.axvspan(df["date"].iloc[i], df["date"].iloc[i] + pd.offsets.MonthEnd(1),
                   color="black", alpha=0.2)

# drawdown line
ax.plot(df["date"], df["drawdown"], color="red", label="Drawdown")

ax.set_title("Stress Regime vs Market Drawdown")
ax.legend()
ax.grid()

plt.savefig(os.path.join(OUT_DIR, "stress_vs_drawdown.png"))
plt.close()