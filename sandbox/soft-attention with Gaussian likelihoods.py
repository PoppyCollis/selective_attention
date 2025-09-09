# Extend the previous script: add Max posterior and Top-2 Difference (standard posterior only)
# on the same graph using a secondary y-axis (still a single figure; no subplots).

import numpy as np
import matplotlib.pyplot as plt

# ---- Helpers ----
def normalize(v, axis=None):
    v = np.asarray(v, dtype=float)
    if axis is None:
        z = v.sum()
        return v / z if z > 0 else np.full_like(v, 1.0 / len(v))
    z = v.sum(axis=axis, keepdims=True)
    out = v / z
    out[np.isnan(out)] = 0.0
    return out


def neg_shannon_entropy(p):
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0.0, p * np.log(p), 0.0)
    return terms.sum()

# def build_p_s_given_phi(M, p_s):
#     """Return p(s|φ) from conditional M."""  
#     # broadcast p(s) across rows  
#     masked = M * p_s # only matters is p_s is not flat
#     return normalize(masked, axis=1)


# def build_p_s_given_phi(G, p_s):
#     masked = G * p_s  # broadcast p(s) across rows
#     return normalize(masked, axis=1)  # rows: φ, cols: s


def gaussian_pdf_2d(x, mean, cov):
    diff = x - mean
    inv_sigma = 1.0 / cov[0, 0]  # isotropic cov
    norm_const = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(cov)))
    quad = -0.5 * inv_sigma * (diff[..., 0] ** 2 + diff[..., 1] ** 2)
    return norm_const * np.exp(quad)


def gaussian_likelihoods(xs, y_const, means, covariances):
    # take all of the 2D xs 
    Xs = np.stack([xs, np.full_like(xs, y_const)], axis=-1)  # [T,2]
    T = Xs.shape[0] # length of sampled batch
    S = means.shape[0] # latent S mean position
    L = np.zeros((T, S), dtype=float) # create empty likelihood array
    # for each of the 3 observation likelihoods
    for s in range(S):
        # calculate the likelihood of all of the datapoints x
        L[:, s] = gaussian_pdf_2d(Xs, means[s], covariances[s])
    return L  # [T,S]


def phi_posterior(p_s_given_phi, p_x_given_s, p_phi_prior):
    m_phi = p_s_given_phi @ p_x_given_s  # [P]
    return normalize(p_phi_prior * m_phi)


def state_posterior_given_phi(p_s_given_phi, p_x_given_s):
    unnorm = p_s_given_phi * p_x_given_s
    return normalize(unnorm, axis=1)  # [P,S] rows over s


# ---- Config ----
config = [-96.0, -59, 96.0]
sigma = 64.0 * 64.0  # variance
means = np.array([[config[0], 0.0],
                  [config[1], 0.0],
                  [config[2], 0.0]], dtype=float)
covariances = np.array([[[sigma, 0.0], [0.0, sigma]],
                        [[sigma, 0.0], [0.0, sigma]],
                        [[sigma, 0.0], [0.0, sigma]]], dtype=float)

epsilon = 0.1
x_min = np.min(means[:, 0]) - epsilon
x_max = np.max(means[:, 0]) + epsilon
xs = np.linspace(x_min, x_max, 201)
y_const = 0.0

# Priors
S = means.shape[0]
p_s = np.full(S, 1.0 / S)  # p(s) flat prior

# --- Define p(s|φ) or a gating mask M with shape [P, S].
# If rows sum ≈1: treated as p(s|φ). Otherwise treated as a mask/weights and
# converted to p(s|φ) via row-normalizing M ⊙ p(s).

P = 2 # size of groups (structural models)

if P == 1:
    M = np.array([[1, 1, 1]])                    # P=1  (φ0: s1,s2,s3)
elif P == 2:
    M = np.array([[1, 1, 0], [0, 1, 1]]) # P=2  (φ0: s1,s2; φ1: s2,s3)
elif P == 3:
    M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # P=3  (φ0: s1; φ1: s2; φ2: s3)
else:
    raise ValueError("Must define 0 < P ≤ S")

# Precompute p(s|φ)
scores = M * p_s 
p_s_given_phi = normalize(scores, axis=1) # only matters is p_s is not flat
print(p_s_given_phi)   

# p_s_given_phi = build_p_s_given_phi(M, p_s)
# print(p_s_given_phi)

p_phi = np.full(P, 1.0 / P)  # p(φ) flat prior

# β values for soft attentio
betas = [0.0, 0.5, 1.0, 5.0, 5, 10, np.inf]


# ---- Compute likelihoods along xs ----
L_xs = gaussian_likelihoods(xs, y_const, means, covariances)  # [T,S]

# Prepare containers for negative entropies and standard metrics
negH_standard = np.zeros_like(xs)
negH_soft = {beta: np.zeros_like(xs) for beta in betas}
max_std = np.zeros_like(xs)
diff_std = np.zeros_like(xs)

# Sweep over x
for t, pxs in enumerate(L_xs):
    # Standard posterior p(s|x) ∝ p(x|s) p(s)
    p_std = normalize(pxs * p_s)
    negH_standard[t] = neg_shannon_entropy(p_std)

    # Max and top-2 difference (standard only)
    # sort descending to get top two
    sorted_p = np.sort(p_std)[::-1]
    max_std[t] = sorted_p[0]
    diff_std[t] = sorted_p[0] - sorted_p[1]

    # Structure posterior p(φ|x)
    p_phi_given_x = phi_posterior(p_s_given_phi, pxs, p_phi)

    # Within-structure posteriors p(s|x,φ)
    p_s_given_x_phi = state_posterior_given_phi(p_s_given_phi, pxs)

    # Soft attention mixture over structures
    for beta in betas:
        if np.isinf(beta):
            w = np.zeros_like(p_phi_given_x)
            w[np.argmax(p_phi_given_x)] = 1.0
        else:
            w = normalize(p_phi_given_x ** beta)
        p_soft = w @ p_s_given_x_phi  # mixture over φ
        negH_soft[beta][t] = neg_shannon_entropy(p_soft)

# ---- Plot on a single figure with twin y-axes ----
fig, ax1 = plt.subplots()

# Left y-axis: negative entropy curves
line_std, = ax1.plot(xs, negH_standard, label="standard neg-entropy")
lines_soft = []
for beta in betas:
    label = r"soft β=∞ neg-entropy" if np.isinf(beta) else f"soft β={beta:g} neg-entropy"
    line, = ax1.plot(xs, negH_soft[beta], label=label)
    lines_soft.append(line)

for m in means[:, 0]:
    ax1.axvline(m, linestyle="--", linewidth=1)

ax1.set_xlabel("horizontal location x")
ax1.set_ylabel("negative Shannon entropy  ∑ p log p")

# Right y-axis: standard Max and Top-2 Difference
ax2 = ax1.twinx()
line_max, = ax2.plot(xs, max_std, label="standard max p", linestyle="-")
line_diff, = ax2.plot(xs, diff_std, label="standard top-2 diff", linestyle=":")
ax2.set_ylabel("posterior max / top-2 diff")

# Combined legend
lines_all = [line_std] + lines_soft + [line_max, line_diff]
labels_all = [l.get_label() for l in lines_all]
ax1.legend(lines_all, labels_all, loc="best")

plt.title("Posterior metrics vs x: neg-entropy (std & soft) + max & top-2 diff (std only)")
plt.show()


def minmax_norm(arr):
    arr = np.array(arr, dtype=float)
    mn, mx = np.min(arr), np.max(arr)
    return (arr - mn) / (mx - mn + 1e-12)

plt.plot(xs, minmax_norm(max_std), label="standard max p", linestyle="-")
plt.plot(xs, minmax_norm(diff_std), label="standard top-2 diff", linestyle=":")
plt.plot(xs, minmax_norm(negH_soft[betas[-1]]), label="beta=inf")
#plt.plot(xs, negH_soft[betas[-1]])
plt.plot(xs, minmax_norm(negH_soft[betas[-2]]), label="beta=1")

plt.legend()
plt.show()

print()