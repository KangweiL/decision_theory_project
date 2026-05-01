import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

# =========================================================
# 0. Paths
# =========================================================
ROOT = "/Users/likangwei/Desktop/DT project"

FEATURE_FP = os.path.join(ROOT, "data_collection", "monthly_regime_features_scaled.csv")
RAW_MONTHLY_FP = os.path.join(ROOT, "data_collection", "monthly_raw_market_regime_data.csv")

OUT_DIR = os.path.join(ROOT, "Final_result_3")
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# 1. User controls
# =========================================================
K = 3

# lower target than before: enough transition to exist, but not dominate
TARGET_TRANSITION_SHARE = 0.12   # try 0.08 to 0.12
MAX_BOOST = 0.05                 # cap on transition boost strength

# weaker persistence adjustment than before
# P_adj = lambda * P_soft + (1-lambda) * I
# larger lambda => weaker identity pull
PERSISTENCE_LAMBDA = 0.9

# EM
MAX_ITER = 250
TOL = 1e-6
RIDGE = 1e-5

# =========================================================
# 2. Load data
# =========================================================
feature_cols = [
    "equity_ret",
    "oil_ret",
    "dollar_ret",
    "vix_level",
    "term_spread",
    "d10_change",
    "trend_12m"
]

df = pd.read_csv(FEATURE_FP)
date_col = "date" if "date" in df.columns else "Date"
df[date_col] = pd.to_datetime(df[date_col]).dt.to_period("M").dt.to_timestamp("M")
df = df.sort_values(date_col).reset_index(drop=True)

df = df[[date_col] + feature_cols].dropna().copy()
X = df[feature_cols].to_numpy()
T, d = X.shape

# =========================================================
# 3. Fit initial 3-regime Gaussian mixture
# =========================================================
gmm = GaussianMixture(
    n_components=K,
    covariance_type="full",
    random_state=42,
    n_init=50
)
gmm.fit(X)

gmm_probs = gmm.predict_proba(X)
gmm_labels = gmm.predict(X)

# Reorder initial regimes:
# high trend + low VIX => calm
# low trend + high VIX => stress
means_gmm = pd.DataFrame(gmm.means_, columns=feature_cols)
score_gmm = means_gmm["trend_12m"] - means_gmm["vix_level"] + 0.25 * means_gmm["equity_ret"]
order0 = list(score_gmm.sort_values(ascending=False).index)   # calm -> transition -> stress

name_map0 = {
    order0[0]: "Calmer Growth",
    order0[1]: "Transition",
    order0[2]: "Stress"
}
color_map0 = {
    order0[0]: "#53C6EA",   # cyan-blue
    order0[1]: "#8F88E8",   # lavender
    order0[2]: "#29E67C"    # green
}

gmm_probs_ord = gmm_probs[:, order0]
initial_shares = gmm_probs_ord.mean(axis=0)

print("Initial GMM regime shares:")
for i, nm in enumerate(["Calmer Growth", "Transition", "Stress"]):
    print(f"  {nm}: {initial_shares[i]:.4f}")

# =========================================================
# 4. Build a moderate transition boost
#    smaller than before, only enough to give transition a real role
# =========================================================
sorted_probs = np.sort(gmm_probs_ord, axis=1)
top1 = sorted_probs[:, -1]
top2 = sorted_probs[:, -2]

ambiguity = 1.0 - (top1 - top2)  # larger => more ambiguous
ambiguity = np.clip(ambiguity, 0.0, 1.0)

trend = df["trend_12m"].to_numpy()
vix = df["vix_level"].to_numpy()
eq = df["equity_ret"].to_numpy()

# favor middle-zone months, not extremes
trend_mid = np.exp(-0.5 * trend**2)
eq_mid = np.exp(-0.5 * eq**2)
vix_mid = np.exp(-0.5 * (vix - 0.25)**2)

transition_score = 0.60 * ambiguity + 0.20 * trend_mid + 0.10 * eq_mid + 0.10 * vix_mid
transition_score = (transition_score - transition_score.min()) / (transition_score.max() - transition_score.min() + 1e-12)

G = gmm_probs_ord.copy()

# choose the smallest boost that gets close to target share
current_transition_share = G[:, 1].mean()
if current_transition_share < TARGET_TRANSITION_SHARE:
    alphas = np.linspace(0.0, MAX_BOOST, 200)
    best_alpha = 0.0
    best_gap = 1e9

    for a in alphas:
        G_try = G.copy()
        G_try[:, 1] = G_try[:, 1] + a * transition_score
        G_try = G_try / G_try.sum(axis=1, keepdims=True)
        gap = abs(G_try[:, 1].mean() - TARGET_TRANSITION_SHARE)
        if gap < best_gap:
            best_gap = gap
            best_alpha = a

    alpha = best_alpha
else:
    alpha = 0.0

G_adj = G.copy()
G_adj[:, 1] = G_adj[:, 1] + alpha * transition_score
G_adj = G_adj / G_adj.sum(axis=1, keepdims=True)

adjusted_shares = G_adj.mean(axis=0)
print("\nAdjusted initial regime shares:")
for i, nm in enumerate(["Calmer Growth", "Transition", "Stress"]):
    print(f"  {nm}: {adjusted_shares[i]:.4f}")
print(f"\nChosen transition boost alpha = {alpha:.4f}")

# =========================================================
# 5. Soft transition matrix initialization
# =========================================================
soft_counts = np.zeros((K, K), dtype=float)
for t in range(1, T):
    soft_counts += np.outer(G_adj[t - 1], G_adj[t])

P_soft = soft_counts / soft_counts.sum(axis=1, keepdims=True)

# weaker persistence adjustment
I = np.eye(K)
P_init = PERSISTENCE_LAMBDA * P_soft + (1.0 - PERSISTENCE_LAMBDA) * I
P_init = P_init / P_init.sum(axis=1, keepdims=True)

print("\nInitial soft transition matrix:")
print(pd.DataFrame(P_init, index=["Calmer Growth","Transition","Stress"], columns=["Calmer Growth","Transition","Stress"]).round(4))

# =========================================================
# 6. Initialize Gaussian HMM emission parameters
# =========================================================
def weighted_mean_cov(X, weights, ridge=1e-5):
    w = weights / (weights.sum() + 1e-12)
    mu = np.sum(X * w[:, None], axis=0)
    xc = X - mu
    Sigma = np.zeros((X.shape[1], X.shape[1]))
    for t in range(X.shape[0]):
        Sigma += w[t] * np.outer(xc[t], xc[t])
    Sigma += ridge * np.eye(X.shape[1])
    return mu, Sigma

mu = np.zeros((K, d))
Sigma = np.zeros((K, d, d))
for k in range(K):
    mu[k], Sigma[k] = weighted_mean_cov(X, G_adj[:, k], ridge=RIDGE)

pi = G_adj[0].copy()
pi = pi / pi.sum()
A = P_init.copy()

# =========================================================
# 7. Gaussian HMM EM
# =========================================================
def compute_log_emission_probs(X, mu, Sigma):
    T = X.shape[0]
    K = mu.shape[0]
    logB = np.zeros((T, K))
    for k in range(K):
        rv = multivariate_normal(mean=mu[k], cov=Sigma[k], allow_singular=False)
        logB[:, k] = rv.logpdf(X)
    return logB

def forward_backward_log(pi, A, logB):
    T, K = logB.shape
    logA = np.log(A + 1e-12)
    logpi = np.log(pi + 1e-12)

    log_alpha = np.zeros((T, K))
    log_alpha[0] = logpi + logB[0]

    for t in range(1, T):
        for j in range(K):
            log_alpha[t, j] = logB[t, j] + logsumexp(log_alpha[t - 1] + logA[:, j])

    loglik = logsumexp(log_alpha[-1])

    log_beta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        for i in range(K):
            log_beta[t, i] = logsumexp(logA[i] + logB[t + 1] + log_beta[t + 1])

    log_gamma = log_alpha + log_beta
    log_gamma = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)

    xi = np.zeros((T - 1, K, K))
    for t in range(T - 1):
        log_xi_t = (
            log_alpha[t][:, None]
            + logA
            + logB[t + 1][None, :]
            + log_beta[t + 1][None, :]
        )
        log_xi_t = log_xi_t - logsumexp(log_xi_t)
        xi[t] = np.exp(log_xi_t)

    return loglik, gamma, xi

def m_step(X, gamma, xi, ridge=1e-5):
    T, d = X.shape
    K = gamma.shape[1]

    pi_new = gamma[0].copy()
    pi_new = pi_new / pi_new.sum()

    A_new = xi.sum(axis=0)
    A_new = A_new / A_new.sum(axis=1, keepdims=True)

    mu_new = np.zeros((K, d))
    Sigma_new = np.zeros((K, d, d))

    for k in range(K):
        w = gamma[:, k]
        wsum = w.sum()
        mu_new[k] = np.sum(X * w[:, None], axis=0) / (wsum + 1e-12)

        xc = X - mu_new[k]
        cov = np.zeros((d, d))
        for t in range(T):
            cov += w[t] * np.outer(xc[t], xc[t])
        cov = cov / (wsum + 1e-12)
        cov += ridge * np.eye(d)
        Sigma_new[k] = cov

    return pi_new, A_new, mu_new, Sigma_new

loglik_history = []
for it in range(MAX_ITER):
    logB = compute_log_emission_probs(X, mu, Sigma)
    loglik, gamma, xi = forward_backward_log(pi, A, logB)
    loglik_history.append(loglik)

    pi_new, A_new, mu_new, Sigma_new = m_step(X, gamma, xi, ridge=RIDGE)

    if it > 0 and abs(loglik_history[-1] - loglik_history[-2]) < TOL:
        print(f"\nEM converged at iteration {it}")
        pi, A, mu, Sigma = pi_new, A_new, mu_new, Sigma_new
        break

    pi, A, mu, Sigma = pi_new, A_new, mu_new, Sigma_new

    if it % 10 == 0:
        print(f"Iteration {it:3d} | loglik = {loglik:.6f}")
else:
    print("\nReached max_iter without hitting tolerance.")

# Final smoothed probabilities with final params
logB = compute_log_emission_probs(X, mu, Sigma)
loglik, gamma, xi = forward_backward_log(pi, A, logB)

# =========================================================
# 8. Final state reorder based on final means
# =========================================================
means_final = pd.DataFrame(mu, columns=feature_cols)
score_final = means_final["trend_12m"] - means_final["vix_level"] + 0.25 * means_final["equity_ret"]
order_final = list(score_final.sort_values(ascending=False).index)  # calm -> transition -> stress

name_map = {
    order_final[0]: "Calmer Growth",
    order_final[1]: "Transition",
    order_final[2]: "Stress"
}
color_map = {
    order_final[0]: "#53C6EA",
    order_final[1]: "#8F88E8",
    order_final[2]: "#29E67C"
}

gamma_ord = gamma[:, order_final]
labels_ord = gamma_ord.argmax(axis=1)

ordered_names = [name_map[i] for i in order_final]
ordered_colors = [color_map[i] for i in order_final]
final_shares = gamma_ord.mean(axis=0) * 100

A_ord = A[np.ix_(order_final, order_final)]
A_ord_df = pd.DataFrame(A_ord, index=ordered_names, columns=ordered_names)

print("\nFinal regime shares:")
for i, nm in enumerate(ordered_names):
    print(f"  {nm}: {final_shares[i]:.2f}%")

print("\nFinal transition matrix:")
print(A_ord_df.round(4))

# =========================================================
# 9. Load raw monthly SP500 if available, else reconstruct from returns
# =========================================================
def find_sp500_column(columns):
    candidates = [
        "sp500", "SP500", "sp500_close", "SP500_Close", "s&p500", "S&P500",
        "s&p_500", "S&P_500", "equity_index", "equity_close", "close"
    ]
    lower_map = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

sp500_df = None
if os.path.exists(RAW_MONTHLY_FP):
    raw_df = pd.read_csv(RAW_MONTHLY_FP)
    raw_date_col = "date" if "date" in raw_df.columns else ("Date" if "Date" in raw_df.columns else None)
    if raw_date_col is not None:
        raw_df[raw_date_col] = pd.to_datetime(raw_df[raw_date_col]).dt.to_period("M").dt.to_timestamp("M")
        sp_col = find_sp500_column(raw_df.columns)
        if sp_col is not None:
            sp500_df = raw_df[[raw_date_col, sp_col]].dropna().copy()
            sp500_df.columns = ["date", "sp500_level"]

if sp500_df is None:
    # fallback: reconstruct a proxy from unscaled cumulative monthly returns if raw index is absent
    # only used if no raw SP500 column is found
    proxy = np.exp(np.cumsum(df["equity_ret"].to_numpy()))
    proxy = 100 * proxy / proxy[0]
    sp500_df = pd.DataFrame({"date": df[date_col].values, "sp500_level": proxy})

plot_df = df[[date_col]].copy()
plot_df.columns = ["date"]
plot_df = plot_df.merge(sp500_df, on="date", how="left").dropna().reset_index(drop=True)

# align gamma to available plot dates
common_dates = pd.Index(df[date_col]).intersection(plot_df["date"])
mask = df[date_col].isin(common_dates).to_numpy()
plot_gamma_ord = gamma_ord[mask]
plot_dates = df.loc[mask, date_col].reset_index(drop=True)
plot_levels = plot_df.set_index("date").loc[plot_dates, "sp500_level"].to_numpy()

sp500_rebased = 100 * plot_levels / plot_levels[0]
plot_labels = plot_gamma_ord.argmax(axis=1)
plot_shares = plot_gamma_ord.mean(axis=0) * 100

starts = plot_dates
ends = plot_dates.shift(-1)
ends.iloc[-1] = plot_dates.iloc[-1] + pd.offsets.MonthEnd(1)

# =========================================================
# 10. Final regime figure
# =========================================================
fig, ax = plt.subplots(figsize=(13.5, 5.5), dpi=150)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for i, (s, e) in enumerate(zip(starts, ends)):
    reg_idx = plot_labels[i]
    ax.axvspan(s, e, facecolor=ordered_colors[reg_idx], alpha=0.82, lw=0)

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
    "Final regime distribution with rebased S&P 500 overlay",
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
plt.savefig(os.path.join(OUT_DIR, "final_regime_distribution_em_v2.png"), bbox_inches="tight")
plt.close()

# =========================================================
# 11. Transition matrix heatmap
# =========================================================
fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=150)
im = ax.imshow(A_ord, cmap="Blues", aspect="equal")

ax.set_xticks(range(K))
ax.set_yticks(range(K))
ax.set_xticklabels(ordered_names, rotation=20, ha="right")
ax.set_yticklabels(ordered_names)

for i in range(K):
    for j in range(K):
        ax.text(j, i, f"{A_ord[i, j]:.3f}", ha="center", va="center", color="black", fontsize=10)

ax.set_title("Estimated Markov Transition Matrix", fontsize=13, weight="bold")
ax.set_xlabel("Next state", fontsize=11)
ax.set_ylabel("Current state", fontsize=11)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "transition_matrix_em_v2.png"), bbox_inches="tight")
plt.close()

# =========================================================
# 12. Save tables
# =========================================================
pd.DataFrame({
    "date": df[date_col],
    "prob_calmer_growth": gamma_ord[:, 0],
    "prob_transition": gamma_ord[:, 1],
    "prob_stress": gamma_ord[:, 2],
    "assigned_regime": [ordered_names[i] for i in gamma_ord.argmax(axis=1)]
}).to_csv(os.path.join(OUT_DIR, "final_regime_probabilities_em_v2.csv"), index=False)

A_ord_df.to_csv(os.path.join(OUT_DIR, "transition_matrix_em_v2.csv"))

pd.DataFrame(mu[order_final], columns=feature_cols, index=ordered_names).to_csv(
    os.path.join(OUT_DIR, "regime_means_em_v2.csv")
)

pd.DataFrame({
    "iteration": np.arange(len(loglik_history)),
    "loglik": loglik_history
}).to_csv(os.path.join(OUT_DIR, "loglik_history_em_v2.csv"), index=False)

print("\nSaved outputs to:")
print(" ", OUT_DIR)
print("  - final_regime_distribution_em_v2.png")
print("  - transition_matrix_em_v2.png")
print("  - final_regime_probabilities_em_v2.csv")
print("  - transition_matrix_em_v2.csv")
print("  - regime_means_em_v2.csv")
print("  - loglik_history_em_v2.csv")