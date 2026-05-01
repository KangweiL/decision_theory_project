import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================
# Paths
# =====================================
ROOT = "/Users/likangwei/Desktop/DT project"

FEATURE_FP = os.path.join(ROOT, "data_collection", "monthly_regime_features_scaled.csv")
RAW_FP = os.path.join(ROOT, "data_collection", "monthly_regime_features.csv")
STATE_FP = os.path.join(ROOT, "Final_result_2", "final_regime_probabilities_em_v2.csv")

OUT_DIR = os.path.join(ROOT, "Final_result_2")

# =====================================
# Load data
# =====================================
df_scaled = pd.read_csv(FEATURE_FP)
df_raw = pd.read_csv(RAW_FP)
df_state = pd.read_csv(STATE_FP)

# align dates
for df in [df_scaled, df_raw, df_state]:
    col = "date" if "date" in df.columns else "Date"
    df[col] = pd.to_datetime(df[col]).dt.to_period("M").dt.to_timestamp("M")

df = df_scaled.merge(df_state, on="date")

# =====================================
# Raw stats (for inverse scaling)
# =====================================
raw_equity = df_raw["equity_ret"]
raw_vix = df_raw["vix_level"]

mu_eq = raw_equity.mean()
std_eq = raw_equity.std()

mu_vix = raw_vix.mean()
std_vix = raw_vix.std()

# =====================================
# Extract regime probabilities
# =====================================
prob_cols = ["prob_calmer_growth", "prob_transition", "prob_stress"]
probs = df[prob_cols].to_numpy()

# =====================================
# Compute emission parameters (scaled)
# =====================================
def weighted_stats(x, weights):
    w = weights / (weights.sum() + 1e-12)
    mean = np.sum(w * x)
    var = np.sum(w * (x - mean)**2)
    return mean, var

results = []

for i, name in enumerate(["Calmer Growth", "Transition", "Stress"]):
    w = probs[:, i]

    # scaled variables
    eq_scaled = df["equity_ret"].to_numpy()
    vix_scaled = df["vix_level"].to_numpy()

    mu_eq_s, var_eq_s = weighted_stats(eq_scaled, w)
    mu_vix_s, var_vix_s = weighted_stats(vix_scaled, w)

    # =====================================
    # Back-transform to original units
    # =====================================
    mu_eq_orig = std_eq * mu_eq_s + mu_eq
    var_eq_orig = (std_eq**2) * var_eq_s

    mu_vix_orig = std_vix * mu_vix_s + mu_vix
    var_vix_orig = (std_vix**2) * var_vix_s

    results.append({
        "state": name,
        "equity_mean": mu_eq_orig,
        "equity_std": np.sqrt(var_eq_orig),
        "vix_mean": mu_vix_orig,
        "vix_std": np.sqrt(var_vix_orig)
    })

df_res = pd.DataFrame(results)
df_res.to_csv(os.path.join(OUT_DIR, "emission_parameters_original_units.csv"), index=False)

print(df_res.round(4))

# =====================================
# Visualization
# =====================================
fig, ax = plt.subplots(1, 2, figsize=(10,4))

# Equity return
ax[0].bar(df_res["state"], df_res["equity_mean"])
ax[0].set_title("Equity Return Mean (Original Units)")
ax[0].set_ylabel("Return")

# VIX
ax[1].bar(df_res["state"], df_res["vix_mean"])
ax[1].set_title("VIX Mean (Original Units)")
ax[1].set_ylabel("Level")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "emission_means_original_units.png"))
plt.show()