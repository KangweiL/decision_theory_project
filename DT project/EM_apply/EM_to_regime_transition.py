import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

# =====================================
# 1. Load the three inputs
# =====================================
features_fp = "/Users/likangwei/Desktop/DT project/data_collection/monthly_regime_features_scaled.csv"
posteriors_fp = "/Users/likangwei/Desktop/DT project/initial_regime_obtain/regime_posteriors_gmm3.csv"
transition_fp = "/Users/likangwei/Desktop/DT project/markov_transition_init/soft_transition_matrix_gmm3_adjusted.csv"

# observed feature data
feature_cols = [
    "equity_ret",
    "oil_ret",
    "dollar_ret",
    "vix_level",
    "term_spread",
    "d10_change",
    "trend_12m"
]

dfX = pd.read_csv(features_fp, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
X = dfX[feature_cols].to_numpy()
T, d = X.shape

# posterior probabilities from GMM initialization
df_post = pd.read_csv(posteriors_fp, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

prob_cols = [
    "prob_calmer_growth",
    "prob_transition",
    "prob_stress"
]

gamma_init = df_post[prob_cols].to_numpy()
K = gamma_init.shape[1]

# initial transition matrix
P_init_df = pd.read_csv(transition_fp, index_col=0)
A_init = P_init_df.to_numpy()

# make sure rows sum to 1
A_init = A_init / A_init.sum(axis=1, keepdims=True)

# =====================================
# 2. Align lengths if needed
# =====================================
# assumes the files share the same monthly sequence
min_T = min(len(dfX), len(df_post))
X = X[:min_T]
gamma_init = gamma_init[:min_T]
T = min_T

# =====================================
# 3. Initialize emission parameters
#    from soft posterior probabilities
# =====================================
def weighted_mean_cov(X, weights, ridge=1e-6):
    """
    X: (T, d)
    weights: (T,)
    """
    w = weights / (weights.sum() + 1e-12)
    mu = np.sum(X * w[:, None], axis=0)

    xc = X - mu
    Sigma = np.zeros((X.shape[1], X.shape[1]))
    for t in range(X.shape[0]):
        Sigma += w[t] * np.outer(xc[t], xc[t])

    Sigma += ridge * np.eye(X.shape[1])
    return mu, Sigma

# initial means and covariances
mu = np.zeros((K, d))
Sigma = np.zeros((K, d, d))

for k in range(K):
    mu[k], Sigma[k] = weighted_mean_cov(X, gamma_init[:, k], ridge=1e-5)

# initial state distribution from first posterior row or sample average
pi = gamma_init[0].copy()
pi = pi / pi.sum()

A = A_init.copy()

# =====================================
# 4. Helper functions for HMM-EM
# =====================================
def compute_log_emission_probs(X, mu, Sigma):
    """
    Returns log B where B[t,k] = log p(X_t | S_t=k)
    """
    T = X.shape[0]
    K = mu.shape[0]
    logB = np.zeros((T, K))

    for k in range(K):
        rv = multivariate_normal(mean=mu[k], cov=Sigma[k], allow_singular=False)
        logB[:, k] = rv.logpdf(X)

    return logB

def forward_backward_log(pi, A, logB):
    """
    Log-space forward-backward.
    Returns:
        loglik
        gamma: smoothed state probs, shape (T, K)
        xi: smoothed pairwise probs, shape (T-1, K, K)
    """
    T, K = logB.shape
    logA = np.log(A + 1e-12)
    logpi = np.log(pi + 1e-12)

    # Forward
    log_alpha = np.zeros((T, K))
    log_alpha[0] = logpi + logB[0]

    for t in range(1, T):
        for j in range(K):
            log_alpha[t, j] = logB[t, j] + logsumexp(log_alpha[t-1] + logA[:, j])

    loglik = logsumexp(log_alpha[T-1])

    # Backward
    log_beta = np.zeros((T, K))
    log_beta[T-1] = 0.0

    for t in range(T-2, -1, -1):
        for i in range(K):
            log_beta[t, i] = logsumexp(logA[i] + logB[t+1] + log_beta[t+1])

    # Smoothed gamma
    log_gamma = log_alpha + log_beta
    log_gamma = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)

    # Smoothed xi
    xi = np.zeros((T-1, K, K))
    for t in range(T-1):
        log_xi_t = (
            log_alpha[t][:, None]
            + logA
            + logB[t+1][None, :]
            + log_beta[t+1][None, :]
        )
        log_xi_t = log_xi_t - logsumexp(log_xi_t)
        xi[t] = np.exp(log_xi_t)

    return loglik, gamma, xi

def m_step(X, gamma, xi, ridge=1e-5):
    """
    Update pi, A, mu, Sigma
    """
    T, d = X.shape
    K = gamma.shape[1]

    # initial state distribution
    pi_new = gamma[0].copy()
    pi_new = pi_new / pi_new.sum()

    # transition matrix
    A_new = xi.sum(axis=0)
    A_new = A_new / A_new.sum(axis=1, keepdims=True)

    # emission parameters
    mu_new = np.zeros((K, d))
    Sigma_new = np.zeros((K, d, d))

    for k in range(K):
        weights = gamma[:, k]
        wsum = weights.sum()

        mu_new[k] = np.sum(X * weights[:, None], axis=0) / (wsum + 1e-12)

        xc = X - mu_new[k]
        cov = np.zeros((d, d))
        for t in range(T):
            cov += weights[t] * np.outer(xc[t], xc[t])

        cov = cov / (wsum + 1e-12)
        cov += ridge * np.eye(d)

        Sigma_new[k] = cov

    return pi_new, A_new, mu_new, Sigma_new

# =====================================
# 5. Run EM
# =====================================
max_iter = 200
tol = 1e-6

loglik_history = []

for iteration in range(max_iter):
    # E-step
    logB = compute_log_emission_probs(X, mu, Sigma)
    loglik, gamma, xi = forward_backward_log(pi, A, logB)
    loglik_history.append(loglik)

    # M-step
    pi_new, A_new, mu_new, Sigma_new = m_step(X, gamma, xi, ridge=1e-5)

    # convergence check
    if iteration > 0 and abs(loglik_history[-1] - loglik_history[-2]) < tol:
        print(f"Converged at iteration {iteration}")
        pi, A, mu, Sigma = pi_new, A_new, mu_new, Sigma_new
        break

    pi, A, mu, Sigma = pi_new, A_new, mu_new, Sigma_new

    if iteration % 10 == 0:
        print(f"Iteration {iteration:3d} | log-likelihood = {loglik:.6f}")

else:
    print("Reached maximum iterations without hitting tolerance.")

# =====================================
# 6. Final state summaries
# =====================================
state_probs = pd.DataFrame(
    gamma,
    columns=["prob_state_1", "prob_state_2", "prob_state_3"]
)

state_probs["date"] = dfX["date"].iloc[:T].values
state_probs["assigned_state"] = gamma.argmax(axis=1) + 1

A_df = pd.DataFrame(
    A,
    index=["state_1", "state_2", "state_3"],
    columns=["state_1", "state_2", "state_3"]
)

pi_df = pd.DataFrame({"state": ["state_1", "state_2", "state_3"], "pi": pi})

mu_df = pd.DataFrame(mu, columns=feature_cols, index=["state_1", "state_2", "state_3"])

print("\nFinal initial-state probabilities:")
print(pi_df)

print("\nEstimated transition matrix:")
print(A_df.round(4))

print("\nEstimated regime means:")
print(mu_df.round(4))

# =====================================
# 7. Save outputs
# =====================================
state_probs.to_csv("ms_em_smoothed_state_probabilities.csv", index=False)
A_df.to_csv("ms_em_transition_matrix.csv")
pi_df.to_csv("ms_em_initial_state_distribution.csv", index=False)
mu_df.to_csv("ms_em_regime_means.csv")

loglik_df = pd.DataFrame({
    "iteration": np.arange(len(loglik_history)),
    "loglik": loglik_history
})
loglik_df.to_csv("ms_em_loglik_history.csv", index=False)