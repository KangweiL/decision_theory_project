import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# =====================================
# 1. Load data
# =====================================
df = pd.read_csv("/Users/likangwei/Desktop/DT project/data_collection/monthly_regime_features_scaled.csv")

feature_cols = [
    "equity_ret",
    "oil_ret",
    "dollar_ret",
    "vix_level",
    "term_spread",
    "d10_change",
    "trend_12m"
]

X = df[feature_cols].dropna().to_numpy()

print("Sample size:", X.shape[0])
print("Number of features:", X.shape[1])

# =====================================
# 2. Compute BIC for different K
# =====================================
K_values = range(1, 6)   # test 1 to 5 regimes
bic_values = []

for K in K_values:
    gmm = GaussianMixture(
        n_components=K,
        covariance_type="full",
        random_state=42,
        n_init=20
    )
    gmm.fit(X)

    bic = gmm.bic(X)
    bic_values.append(bic)

    print(f"K = {K}, BIC = {bic:.4f}")

# =====================================
# 3. Select best K
# =====================================
best_idx = int(np.argmin(bic_values))
best_K = list(K_values)[best_idx]

print("\nBest number of regimes (lowest BIC):", best_K)
print("Lowest BIC value:", bic_values[best_idx])

# =====================================
# 4. Plot BIC
# =====================================
plt.figure(figsize=(7, 4))
plt.plot(list(K_values), bic_values, marker="o")
plt.xlabel("Number of regimes K")
plt.ylabel("BIC")
plt.title("BIC for regime selection (Gaussian Mixture)")
plt.xticks(list(K_values))
plt.grid(True, alpha=0.3)
plt.show()