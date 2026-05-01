import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.mixture import GaussianMixture
import yfinance as yf

# =========================================================
# 0. Paths
# =========================================================
ROOT = "/Users/likangwei/Desktop/DT project"

FEATURE_FP = os.path.join(ROOT, "data_collection", "monthly_regime_features_scaled.csv")
OUT_DIR = os.path.join(ROOT, "initial_regime_2")
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# 1. User controls
# =========================================================
K = 3

# Adjust this manually if you want more / less transition in the initial plot
TARGET_TRANSITION_SHARE = 0.08   # good starting range: 0.06 to 0.12
MAX_BOOST = 0.12                 # cap on transition adjustment

# =========================================================
# 2. Load data
# =========================================================
df = pd.read_csv(FEATURE_FP)
date_col = "date" if "date" in df.columns else "Date"
df[date_col] = pd.to_datetime(df[date_col]).dt.to_period("M").dt.to_timestamp("M")
df = df.sort_values(date_col).reset_index(drop=True)

feature_cols = [
    "equity_ret",
    "oil_ret",
    "dollar_ret",
    "vix_level",
    "term_spread",
    "d10_change",
    "trend_12m"
]

df = df[[date_col] + feature_cols].dropna().copy()
X = df[feature_cols].to_numpy()

# =========================================================
# 3. Fit 3-regime Gaussian mixture
# =========================================================
gmm = GaussianMixture(
    n_components=K,
    covariance_type="full",
    random_state=42,
    n_init=50
)
gmm.fit(X)

probs = gmm.predict_proba(X)
labels = gmm.predict(X)

# =========================================================
# 4. Reorder regimes into economic order
#    calm -> transition -> stress
# =========================================================
means = pd.DataFrame(gmm.means_, columns=feature_cols)
score = means["trend_12m"] - means["vix_level"] + 0.25 * means["equity_ret"]
order = list(score.sort_values(ascending=False).index)

name_map = {
    order[0]: "Calmer Growth",
    order[1]: "Transition",
    order[2]: "Stress"
}
color_map = {
    order[0]: "#53C6EA",   # cyan-blue
    order[1]: "#8F88E8",   # lavender
    order[2]: "#29E67C"    # green
}

probs_ord = probs[:, order]

# =========================================================
# 5. Moderate transition-state adjustment
#    This is the missing step in the old initial plot
# =========================================================
sorted_probs = np.sort(probs_ord, axis=1)
top1 = sorted_probs[:, -1]
top2 = sorted_probs[:, -2]

ambiguity = 1.0 - (top1 - top2)

trend = df["trend_12m"].to_numpy()
vix = df["vix_level"].to_numpy()
eq = df["equity_ret"].to_numpy()

trend_mid = np.exp(-0.5 * trend**2)
eq_mid = np.exp(-0.5 * eq**2)
vix_mid = np.exp(-0.5 * (vix - 0.25)**2)

transition_score = 0.60 * ambiguity + 0.20 * trend_mid + 0.10 * eq_mid + 0.10 * vix_mid
transition_score = (transition_score - transition_score.min()) / (transition_score.max() - transition_score.min() + 1e-12)

current_transition_share = probs_ord[:, 1].mean()

if current_transition_share < TARGET_TRANSITION_SHARE:
    alphas = np.linspace(0.0, MAX_BOOST, 200)
    best_alpha = 0.0
    best_gap = 1e9

    for a in alphas:
        G_try = probs_ord.copy()
        G_try[:, 1] = G_try[:, 1] + a * transition_score
        G_try = G_try / G_try.sum(axis=1, keepdims=True)
        gap = abs(G_try[:, 1].mean() - TARGET_TRANSITION_SHARE)
        if gap < best_gap:
            best_gap = gap
            best_alpha = a

    alpha = best_alpha
else:
    alpha = 0.0

probs_adj = probs_ord.copy()
probs_adj[:, 1] = probs_adj[:, 1] + alpha * transition_score
probs_adj = probs_adj / probs_adj.sum(axis=1, keepdims=True)

labels_adj = probs_adj.argmax(axis=1)

ordered_names = ["Calmer Growth", "Transition", "Stress"]
ordered_colors = ["#53C6EA", "#8F88E8", "#29E67C"]
shares_adj = probs_adj.mean(axis=0) * 100

print("Original initial shares:")
for i, nm in enumerate(ordered_names):
    print(f"  {nm}: {probs_ord[:, i].mean():.4f}")

print("\nAdjusted initial shares:")
for i, nm in enumerate(ordered_names):
    print(f"  {nm}: {probs_adj[:, i].mean():.4f}")

print(f"\nChosen transition boost alpha = {alpha:.4f}")

# =========================================================
# 6. Download real S&P 500 and rebase
# =========================================================
start_date = df[date_col].min().strftime("%Y-%m-%d")
end_date = (df[date_col].max() + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")

sp500 = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=False)
sp500_monthly = sp500[["Close"]].resample("M").last().dropna().reset_index()
sp500_monthly.columns = [date_col, "sp500_close"]
sp500_monthly[date_col] = pd.to_datetime(sp500_monthly[date_col]).dt.to_period("M").dt.to_timestamp("M")

plot_df = pd.merge(df[[date_col]], sp500_monthly[[date_col, "sp500_close"]], on=date_col, how="inner")
common_dates = pd.Index(df[date_col]).intersection(plot_df[date_col])
mask = df[date_col].isin(common_dates).to_numpy()

plot_dates = df.loc[mask, date_col].reset_index(drop=True)
plot_probs = probs_adj[mask]
plot_labels = plot_probs.argmax(axis=1)
plot_shares = plot_probs.mean(axis=0) * 100

plot_levels = plot_df.set_index(date_col).loc[plot_dates, "sp500_close"].to_numpy()
sp500_rebased = 100 * plot_levels / plot_levels[0]

starts = plot_dates
ends = plot_dates.shift(-1)
ends.iloc[-1] = plot_dates.iloc[-1] + pd.offsets.MonthEnd(1)

# =========================================================
# 7. Plot the NEW initial regime figure
# =========================================================
fig, ax = plt.subplots(figsize=(13.5, 5.5), dpi=150)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for i, (s, e) in enumerate(zip(starts, ends)):
    ax.axvspan(s, e, facecolor=ordered_colors[plot_labels[i]], alpha=0.82, lw=0)

ax2 = ax.twinx()
ax2.plot(plot_dates, sp500_rebased, color="#1738F5", linewidth=2.8, solid_capstyle="round")

ax.set_xlim(plot_dates.iloc[0], ends.iloc[-1])
ax.set_ylim(0, 1)
ax.set_yticks([])

for spine in ["left", "right", "top"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#555555")

ax2.spines["top"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.spines["right"].set_color("#555555")
ax2.tick_params(axis="y", colors="#0A0F45", labelsize=10)
ax2.set_ylabel("Rebased level", fontsize=10, color="#0A0F45", weight="bold")

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator())

ax.tick_params(axis="x", which="major", labelsize=10, colors="#0A0F45", length=0)
ax.tick_params(axis="x", which="minor", length=3, color="#888888")

for tick in ax.get_xticklabels():
    tick.set_fontweight("bold")

ax.text(
    0.985, 1.01,
    "Adjusted initial regime distribution with rebased S&P 500 overlay",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=13,
    weight="bold",
    color="#0A0F45"
)

handles = [
    Patch(facecolor=ordered_colors[i], edgecolor=ordered_colors[i],
          label=f"{ordered_names[i]} [{plot_shares[i]:.2f}%]")
    for i in range(K)
]
handles.append(Line2D([0], [0], color="#1738F5", lw=3, label="Rebased S&P 500"))

legend = ax.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
    ncol=2,
    frameon=False,
    fontsize=10,
    handlelength=1.4,
    columnspacing=1.8,
    handletextpad=0.6
)
for txt in legend.get_texts():
    txt.set_color("#2E3B2E")

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(os.path.join(OUT_DIR, "regime_distribution_gmm3_adjusted.png"), bbox_inches="tight")
plt.close()

# =========================================================
# 8. Save adjusted initial posterior and means
# =========================================================
posterior = df[[date_col]].copy()
posterior["prob_calmer_growth"] = probs_adj[:, 0]
posterior["prob_transition"] = probs_adj[:, 1]
posterior["prob_stress"] = probs_adj[:, 2]
posterior["assigned_regime"] = [ordered_names[i] for i in labels_adj]
posterior.to_csv(os.path.join(OUT_DIR, "regime_posteriors_gmm3_adjusted.csv"), index=False)

means_out = means.loc[order].copy()
means_out.index = ordered_names
means_out.to_csv(os.path.join(OUT_DIR, "regime_component_means_gmm3_adjusted.csv"))

print("\nSaved adjusted initial regime outputs to:")
print(" ", os.path.join(OUT_DIR, "regime_distribution_gmm3_adjusted.png"))
print(" ", os.path.join(OUT_DIR, "regime_posteriors_gmm3_adjusted.csv"))
print(" ", os.path.join(OUT_DIR, "regime_component_means_gmm3_adjusted.csv"))