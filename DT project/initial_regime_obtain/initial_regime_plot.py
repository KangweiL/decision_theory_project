import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.mixture import GaussianMixture
import yfinance as yf

# Load data
fp = '/Users/likangwei/Desktop/DT project/data_collection/monthly_regime_features_scaled.csv'
df = pd.read_csv(fp, parse_dates=['date']).sort_values('date').reset_index(drop=True)
feature_cols = ['equity_ret','oil_ret','dollar_ret','vix_level','term_spread','d10_change','trend_12m']
X = df[feature_cols].to_numpy()

# Fit 3-regime Gaussian mixture
K = 3
gmm = GaussianMixture(n_components=K, covariance_type='full', random_state=42, n_init=50)
gmm.fit(X)
probs = gmm.predict_proba(X)
labels = gmm.predict(X)

# Reorder regimes into an interpretable order using a simple score
# higher trend and lower VIX -> calmer regime; lower trend and higher VIX -> stressier regime
means = pd.DataFrame(gmm.means_, columns=feature_cols)
score = means['trend_12m'] - means['vix_level'] + 0.25 * means['equity_ret']
order = list(score.sort_values(ascending=False).index)  # calm -> middle -> stress
name_map = {
    order[0]: 'Calmer Growth',
    order[1]: 'Transition',
    order[2]: 'Stress'
}
color_map = {
    order[0]: '#53C6EA',   # cyan-blue
    order[1]: '#8F88E8',   # lavender
    order[2]: '#29E67C'    # green
}
new_index = {old:i for i, old in enumerate(order)}
labels_ord = np.array([new_index[x] for x in labels])
probs_ord = probs[:, order]

ordered_names = [name_map[i] for i in order]
ordered_colors = [color_map[i] for i in order]
shares = probs_ord.mean(axis=0) * 100

# Build a real rebased S&P 500 index at month-end dates
dates = df['date']

sp500 = yf.download("^GSPC", start="2017-05-01", end="2026-05-01", auto_adjust=False)
sp500_monthly = sp500[["Close"]].resample("M").last().dropna().reset_index()
sp500_monthly.columns = ["date", "sp500_close"]

# align timestamps to month end
sp500_monthly["date"] = pd.to_datetime(sp500_monthly["date"]).dt.to_period("M").dt.to_timestamp("M")
df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp("M")

# merge so the plotted line uses exactly the same timeline as the regime bars
df = df.merge(sp500_monthly[["date", "sp500_close"]], on="date", how="left")
df = df.dropna(subset=["sp500_close"]).reset_index(drop=True)

# if rows were dropped, keep labels/probabilities aligned
X = df[feature_cols].to_numpy()
probs = gmm.predict_proba(X)
labels = gmm.predict(X)
labels_ord = np.array([new_index[x] for x in labels])
probs_ord = probs[:, order]

proxy = 100 * df["sp500_close"] / df["sp500_close"].iloc[0]

dates = df["date"]
starts = dates
ends = dates.shift(-1)
ends.iloc[-1] = dates.iloc[-1] + pd.offsets.MonthEnd(1)

fig, ax = plt.subplots(figsize=(13.5, 5.5), dpi=150)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Background regime spans
for i, (s, e) in enumerate(zip(starts, ends)):
    # use hard assignment for initial plot
    reg_old = order[labels_ord[i]]
    ax.axvspan(s, e, facecolor=color_map[reg_old], alpha=0.82, lw=0)

# Overlay proxy line on twin axis
ax2 = ax.twinx()
ax2.plot(dates, proxy, color='#1738F5', linewidth=2.8, solid_capstyle='round', label='Rebased S&P 500')

# Formatting
ax.set_xlim(dates.iloc[0], ends.iloc[-1])
ax.set_ylim(0, 1)
ax.set_yticks([])
for spine in ['left','right','top']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_color('#555555')

ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_color('#555555')
ax2.tick_params(axis='y', colors='#0A0F45', labelsize=10)
ax2.set_ylabel('Rebased level', fontsize=10, color='#0A0F45', weight='bold')

# Set x ticks to match style
# Major ticks: every 6 months
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

# Minor ticks: every month (no labels, just ticks)
ax.xaxis.set_minor_locator(mdates.MonthLocator())

ax.tick_params(axis='x', which='major', labelsize=10, colors='#0A0F45', length=0)
ax.tick_params(axis='x', which='minor', length=3, color='#888888')
ax.tick_params(axis='x', labelsize=10, colors='#0A0F45', length=0)
for tick in ax.get_xticklabels():
    tick.set_fontweight('bold')

# Title-like annotation
ax.text(0.985, 1.01, 'Regime distribution with rebased S&P 500 overlay',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=13, weight='bold', color='#0A0F45')

# Legend with shares
handles = [
    Patch(facecolor=ordered_colors[i], edgecolor=ordered_colors[i], label=f'{ordered_names[i]} [{shares[i]:.2f}%]')
    for i in range(K)
]
handles.append(Line2D([0], [0], color='#1738F5', lw=3, label='Rebased S&P 500'))
legend = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.16),
                   ncol=2, frameon=False, fontsize=10, handlelength=1.4, columnspacing=1.8,
                   handletextpad=0.6)
for txt in legend.get_texts():
    txt.set_color('#2E3B2E')

plt.tight_layout(rect=[0, 0.08, 1, 1])
out = '/Users/likangwei/Desktop/DT project/regime_distribution_gmm3.png'
plt.savefig(out, bbox_inches='tight')
plt.close()

# Save posterior probabilities and summary
posterior = df[['date']].copy()
for i in range(K):
    posterior[f'prob_{ordered_names[i].lower().replace(" ","_")}'] = probs_ord[:, i]
posterior['assigned_regime'] = [ordered_names[i] for i in labels_ord]
posterior.to_csv('/Users/likangwei/Desktop/DT project/regime_posteriors_gmm3.csv', index=False)

means_out = means.loc[order].copy()
means_out.index = ordered_names
means_out.to_csv('/Users/likangwei/Desktop/DT project/regime_component_means_gmm3.csv')

print('Saved plot to', out)
print('Saved posterior CSV and LaTeX snippet.')
print('Order:', ordered_names)
print('Shares:', shares)