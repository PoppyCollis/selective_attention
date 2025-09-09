# jgen_S5.py — |S|=5, flexible P∈{1..5}, coherent structural priors, tempered attention α

import numpy as np

EPS = 1e-16

# ---------- helpers ----------
def normalize(v, axis=None):
    v = np.asarray(v, float)
    if axis is None:
        z = v.sum()
        return v / (z if z > 0 else 1.0)
    z = v.sum(axis=axis, keepdims=True)
    z = np.where(z > 0, z, 1.0)
    return v / z

def xlogy(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((x > 0) & (y > 0), x * np.log(y), 0.0)

def gaussian_pdf_2d(x, mean, cov):
    diff = x - mean
    det = np.linalg.det(cov)
    inv_sigma = 1.0 / cov[0, 0]  # isotropic
    norm_const = 1.0 / (2.0 * np.pi * np.sqrt(det))
    quad = -0.5 * inv_sigma * (diff[..., 0] ** 2 + diff[..., 1] ** 2)
    return norm_const * np.exp(quad)

def gaussian_likelihoods(xs, y_const, means, covariances):
    Xs = np.stack([xs, np.full_like(xs, y_const)], axis=-1)  # [X,2]
    X, S = Xs.shape[0], means.shape[0]
    L = np.zeros((X, S), float)
    for s in range(S):
        L[:, s] = gaussian_pdf_2d(Xs, means[s], covariances[s])
    return L

def temper_structure_posterior(p_phi_x, alpha):
    p = np.asarray(p_phi_x, float)
    if np.isinf(alpha):
        out = np.zeros_like(p)
        arg = np.argmax(p, axis=1)
        out[np.arange(p.shape[0]), arg] = 1.0
        return out
    return normalize((p + EPS) ** alpha, axis=1)

# ---------- “coherent” structural priors from a mask ----------
def coherent_structural_priors(M, p_s, leak=0.0):
    """
    Build p(φ) and p(s|φ) from a mask M[P,S] so that sum_φ p(φ) p(s|φ) = p(s).
    leak: small ε (e.g. 1e-6) avoids hard zeros if desired.
    """
    M = np.asarray(M, float)
    p_s = normalize(p_s)
    Mtil = leak + (1.0 - leak) * M
    cover = Mtil.sum(axis=0)  # [S]
    if np.any(cover == 0):
        raise ValueError("Some states have no covering structure (a zero column in M).")
    shares = Mtil * (p_s / cover)[None, :]  # [P,S]
    p_phi = shares.sum(axis=1)               # [P], sums to 1
    p_s_phi = shares / p_phi[:, None]        # [P,S]
    return p_phi, p_s_phi

# ---------- joint & MI ----------
def build_joint(p_x, p_phi_x, p_s_xphi):
    if not np.isclose(p_x.sum(), 1.0): raise ValueError("p(x) must sum to 1")
    if not np.allclose(p_phi_x.sum(axis=1), 1.0): raise ValueError("rows of p(φ|x) must sum to 1")
    if not np.allclose(p_s_xphi.sum(axis=2), 1.0): raise ValueError("p(s|x,φ) must sum to 1 over s")
    joint = p_x[:, None, None] * p_phi_x[:, :, None] * p_s_xphi
    Z = joint.sum()
    return joint / Z if not np.isclose(Z, 1.0) else joint

def induced_marginals(joint):
    p_x     = joint.sum(axis=(1,2))
    p_phi   = joint.sum(axis=(0,2))
    p_s     = joint.sum(axis=(0,1))
    p_x_phi = joint.sum(axis=2)
    p_phi_s = joint.sum(axis=0)
    p_s_phi = normalize(p_phi_s, axis=1)
    return dict(p_x=p_x, p_phi=p_phi, p_s=p_s, p_x_phi=p_x_phi, p_phi_s=p_phi_s, p_s_phi=p_s_phi)

def mutual_infos_from_joint(joint, marginals, p_phi_x, p_s_xphi):
    p_x_phi = marginals["p_x_phi"]
    p_phi   = marginals["p_phi"]
    p_phi_s = marginals["p_phi_s"]
    p_s     = marginals["p_s"]
    p_s_phi = marginals["p_s_phi"]

    I_x_phi = xlogy(p_x_phi, (p_phi_x + EPS) / (p_phi[None, :] + EPS)).sum()
    I_phi_s = xlogy(p_phi_s, (p_s_phi + EPS) / (p_s[None, :] + EPS)).sum()
    ratio   = (p_s_xphi + EPS) / (p_s_phi[None, :, :] + EPS)
    I_x_s_phi = xlogy(joint, ratio).sum()
    return I_x_phi, I_phi_s, I_x_s_phi

def expected_utility(joint, U_xs):
    X, _, S = joint.shape
    if U_xs.shape != (X, S): raise ValueError(f"U_xs must be [X,S]={X,S}")
    return (joint * U_xs[:, None, :]).sum()

# ---------- conditionals from mask & Gaussians (with coherent priors) ----------
def build_conditionals_from_mask_and_gaussians(xs, y_const, means, covs, M, p_s=None, leak=0.0):
    S = means.shape[0]
    if p_s is None:
        p_s = np.full(S, 1.0 / S)
    # coherent p(φ), p(s|φ)
    p_phi_prior, p_s_phi = coherent_structural_priors(M, p_s, leak=leak)  # [P], [P,S]
    print(p_s_phi)
    P = p_s_phi.shape[0]

    # p(x) uniform over the sampled xs
    X = len(xs)
    p_x = np.full(X, 1.0 / X)

    # Gaussian likelihoods
    L = gaussian_likelihoods(xs, y_const, means, covs)  # [X,S]

    # p(φ|x) ∝ p(φ) * m_φ(x),  m_φ = Σ_s p(x|s) p(s|φ)
    m = L @ p_s_phi.T                     # [X,P]
    p_phi_x = normalize(p_phi_prior[None, :] * m, axis=1)

    # p(s|x,φ) ∝ p(x|s) p(s|φ), rowwise normalize over s
    p_s_xphi = normalize(p_s_phi[None, :, :] * L[:, None, :], axis=2)  # [X,P,S]

    return p_x, p_phi_x, p_s_xphi, p_phi_prior, p_s_phi, L

# ---------- J_gen (with α on structures, optional γ on p(s|x,φ)) ----------
def J_gen(p_x, p_phi_x, p_s_xphi, U_xs, beta1, beta2, beta3, alpha=1.0, gamma=1.0, return_parts=False):
    p_phi_x_eff = temper_structure_posterior(p_phi_x, alpha)
    p_s_xphi_eff = p_s_xphi if gamma == 1.0 else normalize((p_s_xphi + EPS) ** gamma, axis=2)

    joint = build_joint(p_x, p_phi_x_eff, p_s_xphi_eff)
    marg = induced_marginals(joint)

    I1, I2, I3 = mutual_infos_from_joint(joint, marg, p_phi_x_eff, p_s_xphi_eff)
    EU = expected_utility(joint, U_xs)

    J = EU - (1.0 / beta1) * I1 - (1.0 / beta2) * I2 - (1.0 / beta3) * I3
    I_eff = min(I1, I2) + I3  # effective throughput in nats

    if not return_parts:
        return J
    return {
        "J_gen": J, "E[U]": EU,
        "I(X;Phi)_nats": I1, "I(Phi;S)_nats": I2, "I(X;S|Phi)_nats": I3,
        "I_eff(X;S)_nats": I_eff,
        "joint": joint, "marginals": marg,
        "p_phi_x_tempered": p_phi_x_eff, "p_s_xphi_tempered": p_s_xphi_eff,
    }

# ---------- default masks for S=5 ----------
def default_mask(S, P):
    if S != 5:
        raise ValueError("This helper is tuned for S=5.")
    M = np.zeros((P, S), float)
    if P == 1:
        M[0, :] = 1
    elif P == 2:
        M[0, 0:3] = 1     # {s1,s2,s3}
        M[1, 2:5] = 1     # {s3,s4,s5}
    elif P == 3:
        M[0, 0:3] = 1     # {s1,s2,s3}
        M[1, 1:4] = 1     # {s2,s3,s4}
        M[2, 2:5] = 1     # {s3,s4,s5}
    elif P == 4:
        M[0, 0:2] = 1     # {s1,s2}
        M[1, 1:3] = 1     # {s2,s3}
        M[2, 2:4] = 1     # {s3,s4}
        M[3, 3:5] = 1     # {s4,s5}
    elif P == 5:
        M = np.eye(5)     # singletons
    else:
        raise ValueError("P must be in {1..5}")
    return M


# ---------- example usage ----------
if __name__ == "__main__":
    # --- Gaussian config (S=5) ---
    S = 5
    config = np.array([-192.0, -96.0, 0.0, 96.0, 192.0])
    sigma = 64.0 * 64.0
    means = np.stack([config, np.zeros_like(config)], axis=1)  # [5,2]
    cov = np.array([[[sigma, 0.0], [0.0, sigma]]] * S)

    # x-grid along horizontal axis
    eps = 0.1
    xs = np.linspace(config.min() - eps, config.max() + eps, 401)
    y_const = 0.0

    # global prior over states
    p_s = np.full(S, 1.0 / S)

    # choose P (1..5) and build a mask
    P = 2
    M = default_mask(S, P)

    # build conditionals with *coherent* p(φ), p(s|φ)
    p_x, p_phi_x, p_s_xphi, p_phi_prior, p_s_phi, L = build_conditionals_from_mask_and_gaussians(
        xs, y_const, means, cov, M, p_s=p_s, leak=0.0
    )

    # Sanity: mixture of structural priors reproduces p_s
    mix_prior = p_phi_prior @ p_s_phi  # [S]
    assert np.allclose(mix_prior, p_s, atol=1e-12), "Coherence check failed: Σφ p(φ)p(s|φ) ≠ p(s)."

    # utility: log-likelihood (proper log score)
    U_xs = np.log(L + EPS)

    # prices & temperatures
    beta1 = beta2 = beta3 = 1.0
    alpha = 1  # 1.0 Bayesian; 0.0 uniform; np.inf hard argmax structure per x
    gamma = 1.0   # optional: temper p(s|x,φ)

    parts = J_gen(p_x, p_phi_x, p_s_xphi, U_xs, beta1, beta2, beta3,
                  alpha=alpha, gamma=gamma, return_parts=True)

    print(f"P={P}, alpha={alpha}")
    print(f"J_gen               : {parts['J_gen']:.6f}")
    print(f"E[U]                : {parts['E[U]']:.6f}")
    print(f"I(X;Phi) [nats]     : {parts['I(X;Phi)_nats']:.6f}")
    print(f"I(Phi;S) [nats]     : {parts['I(Phi;S)_nats']:.6f}")
    print(f"I(X;S|Phi) [nats]   : {parts['I(X;S|Phi)_nats']:.6f}")
    print(f"I_eff(X;S) [nats]   : {parts['I_eff(X;S)_nats']:.6f}")
