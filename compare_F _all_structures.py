# Extend the previous solver so it can take a *data-conditioned* world distribution p(w|x)
# computed from a Gaussian likelihood model with mus, sigmas and a given observation x.

from itertools import combinations
import numpy as np
import pandas as pd

def softmax(logits, axis=-1):
    logits = np.asarray(logits)
    m = np.max(logits, axis=axis, keepdims=True)
    x = np.exp(logits - m)
    return x / np.sum(x, axis=axis, keepdims=True)

def kl_div(q, p, eps=1e-12):
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    q = np.clip(q, eps, 1.0)
    p = np.clip(p, eps, 1.0)
    return np.sum(q * (np.log(q) - np.log(p)))

def generate_p_w_phis(k, n_hot=None):
    if n_hot is None:
        ns = list(range(1, k+1))
    elif isinstance(n_hot, int):
        ns = [n_hot]
    else:
        ns = list(n_hot)

    rows = []
    for n in ns:
        for ones_indices in combinations(range(k), n):
            vec = np.zeros(k, dtype=int)
            vec[list(ones_indices)] = 1
            rows.append(vec)
    return np.array(rows, dtype=int)

def expected_utility_matrix(U, p_w, p_phi_w, p_a_w_phi):
    k = U.shape[0]
    num_phi = p_phi_w.shape[1]
    EU = 0.0
    for w in range(k):
        for j in range(num_phi):
            EU += p_w[w] * p_phi_w[w, j] * np.dot(p_a_w_phi[w, j, :], U[w, :])
    return EU

def run_parallel_bounded_rational(
    U, p_w,
    beta1=1.0, beta3=1.0, 
    n_hot=2, max_iter=1000, tol=1e-9, seed=0
):
    """
    Parallel hierarchy (Eqs. 26–29) with *data-conditioned* p_w (e.g., p(w|x)).
    """
    rng = np.random.default_rng(seed)
    k = U.shape[0]
    assert U.shape == (k, k)

    # Build phi masks (each phi selects subset of states)
    masks = generate_p_w_phis(k, n_hot=n_hot)  # (num_phi, k)
    num_phi = masks.shape[0]

    # Initialize p(phi): uniform
    p_phi = np.ones(num_phi) / num_phi

    # Initialize p(phi|w): proportional to p(phi) (softmax over phi per w)
    p_phi_w = np.tile(p_phi, (k, 1))

    # Initialize p(a|phi): uniform over actions
    p_a_phi = np.ones((num_phi, k)) / k

    # Initialize p(a|w,phi): softmax of log p(a|phi) + small noise
    noise = rng.normal(scale=1e-3, size=(k, num_phi, k))
    logits_init = np.log(p_a_phi)[None, :, :] + noise
    p_a_w_phi = softmax(logits_init, axis=-1)

    history = {"EU": [], "I(W;Phi)": [], "max_delta": []}

    def compute_I_WPhi(p_w, p_phi, p_phi_w):
        val = 0.0
        eps = 1e-12
        for w in range(k):
            for j in range(num_phi):
                q = max(p_phi_w[w, j], eps)
                p = max(p_phi[j], eps)
                val += p_w[w] * q * (np.log(q) - np.log(p))
        return val

    last_arrays = None

    for it in range(max_iter):
        # (1) p*(a|w,phi) ∝ p*(a|phi) * exp(beta3 * U(w,a))
        logits = np.log(p_a_phi)[None, :, :] + beta3 * U[:, None, :]
        p_a_w_phi_new = softmax(logits, axis=-1)

        # (2) ΔF_parallel(w,phi) = E_{p(a|w,phi)}[U] - (1/beta3) KL( p(a|w,phi) || p(a|phi) )
        DeltaF = np.zeros((k, num_phi))
        for w in range(k):
            for j in range(num_phi):
                EU = np.dot(p_a_w_phi_new[w, j, :], U[w, :])
                KL = kl_div(p_a_w_phi_new[w, j, :], p_a_phi[j, :])
                DeltaF[w, j] = EU - (1.0 / beta3) * KL

        # (3) p*(phi|w) ∝ p(phi) * exp(beta1 * ΔF(w,phi))
        logits_phi_w = np.log(p_phi)[None, :] + beta1 * DeltaF
        p_phi_w_new = softmax(logits_phi_w, axis=1)  # across phi

        # (4) p(phi) = sum_w p(w|x) p*(phi|w)   [replace prior p(w) by posterior p(w|x)]
        p_phi_new = np.einsum('w,wj->j', p_w, p_phi_w_new)
        p_phi_new = p_phi_new / np.sum(p_phi_new)

        # (5) p(w|phi,x) ∝ p*(phi|w) p(w|x)
        p_w_phi = (p_phi_w_new * p_w[:, None]) / np.maximum(p_phi_new[None, :], 1e-12)
        p_w_phi /= np.maximum(np.sum(p_w_phi, axis=0, keepdims=True), 1e-12)

        # (6) p*(a|phi) = sum_w p(w|phi,x) p*(a|w,phi)
        p_a_phi_new = np.einsum('wj,wja->ja', p_w_phi, p_a_w_phi_new)
        p_a_phi_new /= np.maximum(np.sum(p_a_phi_new, axis=1, keepdims=True), 1e-12)

        arrays_now = (p_a_w_phi_new, p_phi_w_new, p_phi_new, p_a_phi_new)
        if last_arrays is None:
            max_delta = np.inf
        else:
            deltas = [np.max(np.abs(A - B)) for A, B in zip(arrays_now, last_arrays)]
            max_delta = float(np.max(deltas))

        # Commit updates
        p_a_w_phi = p_a_w_phi_new
        p_phi_w   = p_phi_w_new
        p_phi     = p_phi_new
        p_a_phi   = p_a_phi_new

        # Track metrics
        EU_overall = expected_utility_matrix(U, p_w, p_phi_w, p_a_w_phi)
        I_WPhi = compute_I_WPhi(p_w, p_phi, p_phi_w)
        history["EU"].append(EU_overall)
        history["I(W;Phi)"].append(I_WPhi)
        history["max_delta"].append(max_delta)

        last_arrays = arrays_now
        if max_delta < tol:
            break

    out = {
        "p_w|x": p_w,
        "masks": masks,
        "p_phi": p_phi,
        "p_phi|w": p_phi_w,
        "p_w|phi,x": p_w_phi,
        "p_a|phi": p_a_phi,
        "p_a|w,phi": p_a_w_phi,
        "history": history,
        "iterations": it + 1,
        "beta1": beta1,
        "beta3": beta3,
    }
    return out

# --- Build p(w|x) from Gaussian likelihoods and a uniform prior ---
def gaussian_loglik(x, mu, sigma):
    return -0.5*np.log(2*np.pi*sigma**2) - 0.5*((x-mu)/sigma)**2

k = 3
mus = np.array([0.0, 1.0, 2.0])    # means
sigmas = np.array([1.0, 1.0, 1.0]) # stds
x_obs = 0.1

# Prior p(w): uniform
p_w_prior = np.ones(k) / k

# Posterior p(w|x) ∝ p(x|w) p(w)
loglik = np.array([gaussian_loglik(x_obs, mus[i], sigmas[i]) for i in range(k)])
log_post = np.log(p_w_prior) + loglik
p_w_post = softmax(log_post, axis=0)

print("Observation x =", x_obs)
print("Gaussian means mus =", mus)
print("Posterior p(w|x):", p_w_post.round(6))

# --- Run the parallel bounded-rational solver using p(w|x) ---
U = np.eye(k)           # 0-1 accuracy utility
beta1 = 3.0             # structure selection sharpness
beta3 = 3.0             # action selection sharpness
res = run_parallel_bounded_rational(U, p_w_post, beta1=beta1, beta3=beta3, n_hot=2, max_iter=500, tol=1e-12, seed=1)

# Pretty-print results
masks = res["masks"]
phi_labels = ["".join(map(str, row)) for row in masks]

print("\n=== Structures (phi) masks (each string marks which w's are ON) ===")
for lbl, row in zip(phi_labels, masks):
    print(lbl, "->", row.tolist())

print("\n=== p(w|x) (data-conditioned prior) ===")
print(p_w_post.round(6))

print("\n=== p(phi) at convergence ===")
df = pd.DataFrame({"phi": phi_labels, "p(phi)": res["p_phi"].round(6)})
print(df.to_string(index=False))

print("\n=== p(phi|w) at convergence ===")
df = pd.DataFrame(res["p_phi|w"], columns=phi_labels, index=[f"w{i+1}" for i in range(k)])
print(df.round(6).to_string())

print("\n=== p(a|phi) at convergence ===")
df = pd.DataFrame(res["p_a|phi"], columns=[f"a{i+1}" for i in range(k)], index=phi_labels)
print(df.round(6).to_string())

print("\n=== Convergence history (last 5 rows) ===")
hist = pd.DataFrame(res["history"])
print(hist.tail().round(6).to_string(index=False))

print("\nIterations:", res["iterations"])
