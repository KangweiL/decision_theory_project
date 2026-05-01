import pandas as pd
import numpy as np

dfX = pd.read_csv("/Users/likangwei/Desktop/DT project/data_collection/monthly_regime_features_scaled.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
dfG = pd.read_csv("/Users/likangwei/Desktop/DT project/regime_posteriors_gmm3_adjusted.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)

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

df = pd.merge(dfX, dfG, on="date", how="inner")

X = df[feature_cols].to_numpy()
G = df[prob_cols].to_numpy()

K = G.shape[1]
d = X.shape[1]

means = np.zeros((K, d))
covs = np.zeros((K, d, d))

for k in range(K):
    w = G[:, k]
    w = w / (w.sum() + 1e-12)

    means[k] = np.sum(X * w[:, None], axis=0)

    xc = X - means[k]
    cov = np.zeros((d, d))
    for t in range(len(X)):
        cov += w[t] * np.outer(xc[t], xc[t])

    cov += 1e-5 * np.eye(d)
    covs[k] = cov

for k, name in enumerate(prob_cols):
    print(f"\nInitial mean for {name}:")
    print(pd.Series(means[k], index=feature_cols).round(4))