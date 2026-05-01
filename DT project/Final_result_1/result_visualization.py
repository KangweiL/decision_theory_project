import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
import yfinance as yf

# =====================================
# 1. Load EM outputs and feature data
# =====================================
features_fp = "/Users/likangwei/Desktop/DT project/data_collection/monthly_regime_features_scaled.csv"
state_fp = "/Users/likangwei/Desktop/DT project/EM_apply/ms_em_smoothed_state_probabilities.csv"
transition_fp = "/Users/likangwei/Desktop/DT project/EM_apply/ms_em_transition_matrix.csv"

feature_cols = [
    "equity_ret",
    "oil_ret",
    "dollar_ret",
    "vix_level",
    "term_spread",
    "d10_change",
    "trend_12m"
]

df_feat = pd.read_csv(features_fp, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
df_state = pd.read_csv(state_fp, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
A = pd.read_csv(transition_fp, index_col=0)

# align dates to month end
df_feat["date"] = pd.to_datetime(df_feat["date"]).dt.to_period("M").dt.to_timestamp("M")
df_state["date"] = pd.to_datetime(df_state["date"]).dt.to_period("M").dt.to_timestamp("M")

# merge
df = pd.merge(df_feat, df_state, on="date", how="inner")

# =====================================
# 2. Use EM smoothed probabilities
# =====================================
prob_cols = ["prob_state_1", "prob_state_2", "prob_state_3"]
probs = df[prob_cols].to_numpy()

# hard assignment from final smoothed probabilities
labels = probs.argmax(axis=1)

# =====================================
# 3. Reorder regimes into interpretable order
#    same idea as before: calmer -> transition -> stress
# =====================================
state_means = []
for k in range(3):
    weights = probs[:, k]
    weights = weights / weights.sum()
    weighted_mean = np.sum(df[feature_cols].to_numpy() * weights[:, None], axis=0)
    state_means.append(weighted_mean)

means = pd.DataFrame(state_means, columns=feature_cols, index=[0, 1, 2])

score = means["trend_12m"] - means["vix_level"] + 0.25 * means["equity_ret"]
order = list(score.sort_values(ascending=False).index)   # calm -> middle -> stress

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

new_index = {old: i for i, old in enumerate(order)}
labels_ord = np.array([new_index[x] for x in labels])
probs_ord = probs[:, order]

ordered_names = [name_map[i] for i in order]
ordered_colors = [color_map[i] for i in order]
shares = probs_ord.mean(axis=0) * 100

# =====================================
# 4. Download and rebase real S&P 500
# =====================================
start_date = df["date"].min().strftime("%Y-%m-%d")
end_date = (df["date"].max() + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")

sp500 = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=False)
sp500_monthly = sp500[["Close"]].resample("M").last().dropna().reset_index()
sp500_monthly.columns = ["date", "sp500_close"]
sp500_monthly["date"] = pd.to_datetime(sp500_monthly["date"]).dt.to_period("M").dt.to_timestamp("M")

df = pd.merge(df, sp500_monthly[["date", "sp500_close"]], on="date", how="inner")
df = df.reset_index(drop=True)

# keep probabilities aligned after merge
probs_ord = probs_ord[:len(df)]
labels_ord = labels_ord[:len(df)]
shares = probs_ord.mean(axis=0) * 100

sp500_rebased = 100 * df["sp500_close"] / df["sp500_close"].iloc[0]

dates = df["date"]
starts = dates
ends = dates.shift(-1)
ends.iloc[-1] = dates.iloc[-1] + pd.offsets.MonthEnd(1)

# =====================================
# 5. Final regime graph
# =====================================
fig, ax = plt.subplots(figsize=(13.5, 5.5), dpi=150)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# background regime spans
for i, (s, e) in enumerate(zip(starts, ends)):
    reg_old = order[labels_ord[i]]
    ax.axvspan(s, e, facecolor=color_map[reg_old], alpha=0.82, lw=0)

# overlay rebased S&P 500
ax2 = ax.twinx()
ax2.plot(
    dates,
    sp500_rebased,
    color="#1738F5",
    linewidth=2.8,
    solid_capstyle="round",
    label="Rebased S&P 500"
)

# formatting
ax.set_xlim(dates.iloc[0], ends.iloc[-1])
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

# finer time axis
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator())

ax.tick_params(axis="x", which="major", labelsize=10, colors="#0A0F45", length=0)
ax.tick_params(axis="x", which="minor", length=3, color="#888888")

for tick in ax.get_xticklabels():
    tick.set_fontweight("bold")

ax.text(
    0.985, 1.01,
    "Final regime distribution with rebased S&P 500 overlay",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=13,
    weight="bold",
    color="#0A0F45"
)

handles = [
    Patch(
        facecolor=ordered_colors[i],
        edgecolor=ordered_colors[i],
        label=f"{ordered_names[i]} [{shares[i]:.2f}%]"
    )
    for i in range(3)
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
plt.savefig("final_regime_distribution_em.png", bbox_inches="tight")
plt.show()

# =====================================
# 6. Reorder transition matrix to match regime order
# =====================================
A_np = A.to_numpy()
A_ord = A_np[np.ix_(order, order)]

A_ord_df = pd.DataFrame(
    A_ord,
    index=ordered_names,
    columns=ordered_names
)

# =====================================
# 7. Transition matrix heatmap
# =====================================
plt.figure(figsize=(6.5, 5.2), dpi=150)
sns.heatmap(
    A_ord_df,
    annot=True,
    fmt=".3f",
    cmap="Blues",
    cbar=True,
    square=True,
    linewidths=0.6,
    linecolor="white"
)
plt.title("Estimated Markov Transition Matrix", fontsize=13, weight="bold")
plt.xlabel("Next state", fontsize=11)
plt.ylabel("Current state", fontsize=11)
plt.tight_layout()
plt.savefig("transition_matrix_em.png", bbox_inches="tight")
plt.show()

# =====================================
# 8. Save ordered outputs
# =====================================
A_ord_df.to_csv("transition_matrix_em_ordered.csv")

final_probs = df[["date"]].copy()
for i in range(3):
    final_probs[f"prob_{ordered_names[i].lower().replace(' ', '_')}"] = probs_ord[:, i]
final_probs["assigned_regime"] = [ordered_names[i] for i in labels_ord]
final_probs.to_csv("final_regime_probabilities_em.csv", index=False)