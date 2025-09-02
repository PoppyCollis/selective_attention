# Bar charts of action posteriors for different beta weightings.
# (Each chart is a separate figure to comply with the "no subplots" rule.)

import numpy as np
import matplotlib.pyplot as plt

# ----- Setup (you can edit these) -----
G = np.array([
    [1, 1, 0],  # phi0 allows s1, s2
    [0, 1, 1],  # phi1 allows s2, s3
], dtype=float)

S = 3
p_s = np.full(S, 1.0 / S)         # p(s)
p_phi = np.full(G.shape[0], 0.5)  # p(phi)
p_x_given_s = np.array([0.2, 0.6, 0.2], dtype=float)  # p(x|s)

betas = [1.0, 2.0, 5.0, np.inf]


def normalize(v, axis=None):
    v = np.asarray(v, dtype=float)
    if axis is None:
        z = v.sum()
        return v / z if z > 0 else np.full_like(v, 1.0 / len(v))
    z = v.sum(axis=axis, keepdims=True)
    out = v / z
    out[np.isnan(out)] = 0.0
    return out


def build_p_s_given_phi(G, p_s):
    masked = G * p_s
    return normalize(masked, axis=1)


def phi_posterior(p_s_given_phi, p_x_given_s, p_phi_prior):
    m_phi = p_s_given_phi @ p_x_given_s  # evidence per structure
    return normalize(p_phi_prior * m_phi)


def state_posterior_given_phi(p_s_given_phi, p_x_given_s):
    unnorm = p_s_given_phi * p_x_given_s
    return normalize(unnorm, axis=1)


# Compute shared terms
p_s_given_phi = build_p_s_given_phi(G, p_s)
p_phi_given_x = phi_posterior(p_s_given_phi, p_x_given_s, p_phi)
p_s_given_x_phi = state_posterior_given_phi(p_s_given_phi, p_x_given_s)

# Create separate bar charts for each beta
for beta in betas:
    if np.isinf(beta):
        w = np.zeros_like(p_phi_given_x)
        w[np.argmax(p_phi_given_x)] = 1.0
    else:
        w = normalize(p_phi_given_x ** beta)
    p_a_given_x_beta = w @ p_s_given_x_phi  # actions map to states

    # Plot
    fig = plt.figure()
    x = np.arange(1, S + 1)
    plt.bar(x, p_a_given_x_beta)
    plt.xticks(x, [f"a={i}" for i in x])
    plt.ylim(0, 1)
    title_beta = "∞" if np.isinf(beta) else f"{beta:g}"
    plt.title(f"Action posterior p(a|x; β={title_beta})")
    plt.ylabel("Probability")

    # Annotate bars
    for xi, yi in zip(x, p_a_given_x_beta):
        plt.text(xi, yi + 0.02, f"{yi:.2f}", ha="center")

    plt.show()
