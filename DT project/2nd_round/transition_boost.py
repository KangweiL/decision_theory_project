import pandas as pd
import numpy as np

# =====================================
# 1. Load feature data and initial posteriors
# =====================================
dfX = pd.read_csv("/Users/likangwei/Desktop/DT project/data_collection/monthly_regime_features_scaled.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
dfP = pd.read_csv("/Users/likangwei/Desktop/DT project/initial_regime_obtain/regime_posteriors_gmm3.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)

feature_cols = [
    "equity_ret",
    "oil_ret",
    "dollar_ret",
    "vix_level",
    "term_spread",
    "d10_change",
    "trend_12m"
]

prob_cols = [
    "prob_calmer_growth",
    "prob_transition",
    "prob_stress"
]

df = pd.merge(dfX, dfP, on="date", how="inner").copy()

G = df[prob_cols].to_numpy()   # initial soft assignments, shape (T,3)

# =====================================
# 2. Build a transition-likeness score
# =====================================
# ambiguity: smaller gap between top two probs => more transitional
sorted_probs = np.sort(G, axis=1)
top1 = sorted_probs[:, -1]
top2 = sorted_probs[:, -2]
ambiguity = 1.0 - (top1 - top2)   # larger = more ambiguous

# use scaled features directly
trend = df["trend_12m"].to_numpy()
vix = df["vix_level"].to_numpy()
eq = df["equity_ret"].to_numpy()

# transition is more likely when:
# - ambiguity is high
# - trend is near the middle (not extreme)
# - VIX is moderate/high but not too extreme
trend_mid = np.exp(-0.5 * trend**2)          # high near 0
vix_mid = np.exp(-0.5 * (vix - 0.5)**2)      # can tune center
eq_mid = np.exp(-0.5 * eq**2)                # avoid extreme return months

transition_score = 0.50 * ambiguity + 0.25 * trend_mid + 0.15 * vix_mid + 0.10 * eq_mid

# normalize to [0,1]
transition_score = (transition_score - transition_score.min()) / (transition_score.max() - transition_score.min() + 1e-12)

# =====================================
# 3. Boost the transition probability
# =====================================
# regime order in your posterior file is:
# 0 = calmer growth, 1 = transition, 2 = stress
G_adj = G.copy()

boost_strength = 0.30   # increase if transition still too small
G_adj[:, 1] = G_adj[:, 1] + boost_strength * transition_score

# renormalize rows
G_adj = G_adj / G_adj.sum(axis=1, keepdims=True)

# inspect new average shares
shares_old = G.mean(axis=0)
shares_new = G_adj.mean(axis=0)

print("Old regime shares:")
print(dict(zip(prob_cols, np.round(shares_old, 4))))

print("\nAdjusted regime shares:")
print(dict(zip(prob_cols, np.round(shares_new, 4))))

# save adjusted soft assignments
df_adj = df[["date"]].copy()
for i, c in enumerate(prob_cols):
    df_adj[c] = G_adj[:, i]
df_adj.to_csv("regime_posteriors_gmm3_adjusted.csv", index=False)