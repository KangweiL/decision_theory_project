import pandas as pd
import numpy as np

# =====================================
# 1. Load posterior probabilities
# =====================================
df = pd.read_csv("/Users/likangwei/Desktop/DT project/regime_posteriors_gmm3_adjusted.csv", parse_dates=["date"])

# Use the posterior probability columns from your file
prob_cols = [
    "prob_calmer_growth",
    "prob_transition",
    "prob_stress"
]

# Keep only needed columns and sort by date
df = df[["date"] + prob_cols].sort_values("date").reset_index(drop=True)

# Convert to numpy array: shape (T, K)
probs = df[prob_cols].to_numpy()
T, K = probs.shape

print("Number of periods:", T)
print("Number of regimes:", K)

# =====================================
# 2. Compute soft transition counts
# =====================================
soft_counts = np.zeros((K, K), dtype=float)

for t in range(1, T):
    soft_counts += np.outer(probs[t - 1], probs[t])

soft_counts_df = pd.DataFrame(
    soft_counts,
    index=prob_cols,
    columns=prob_cols
)

print("\nSoft transition counts:")
print(soft_counts_df.round(4))

# =====================================
# 3. Row-normalize to get initial transition matrix
# =====================================
row_sums = soft_counts.sum(axis=1, keepdims=True)
P_init_soft = soft_counts / row_sums

P_init_soft_df = pd.DataFrame(
    P_init_soft,
    index=prob_cols,
    columns=prob_cols
)

print("\nInitial soft transition matrix:")
print(P_init_soft_df.round(4))

# =====================================
# 4. Mild persistence adjustment
#    Blend with identity matrix
# =====================================
lambda_ = 0.9   # keep 90% of empirical matrix, add 10% persistence
I = np.eye(K)

P_init_adjusted = lambda_ * P_init_soft + (1 - lambda_) * I

# normalize again to avoid tiny rounding issues
P_init_adjusted = P_init_adjusted / P_init_adjusted.sum(axis=1, keepdims=True)

P_init_adjusted_df = pd.DataFrame(
    P_init_adjusted,
    index=prob_cols,
    columns=prob_cols
)

print("\nPersistence-adjusted initial transition matrix:")
print(P_init_adjusted_df.round(4))

# =====================================
# 5. Optional: save results
# =====================================
soft_counts_df.to_csv("soft_transition_counts_gmm3.csv")
P_init_soft_df.to_csv("soft_transition_matrix_gmm3.csv")
P_init_adjusted_df.to_csv("soft_transition_matrix_gmm3_adjusted.csv")