"""
Free energy with *predefined* priors vs J_gen (optimal/self-consistent priors)

Given:
  - p_x        : [X]                 prior over x
  - p_phi_x    : [X,P]               p(φ | x)
  - p_s_xphi   : [X,P,S]             p(s | x, φ)
  - U_xs       : [X,S] or None       utility U(x,s); if None, E[U]=0
  - betas      : (β1, β2, β3)

Predefined priors (baselines) for FE:
  - r_phi      : [P]                 baseline for φ   (i.e., prior P(φ))
  - r_s        : [S]                 baseline for s   (i.e., prior p(s))
  - r_s_phi    : [P,S]               baseline for s|φ (i.e., prior p(s|φ))

This computes:
  - F_fixed    = E[U] - (1/β1) E_x KL(p(φ|x) || r_phi)
                        - (1/β2) E_φ KL(p(s|φ) || r_s)
                        - (1/β3) E_{x,φ} KL(p(s|x,φ) || r_s_phi[φ])
  - J_gen      = E[U] - (1/β1) I(X;Φ) - (1/β2) I(Φ;S) - (1/β3) I(X;S|Φ)
                 (computed from the SAME joint; “optimal priors”)
  - Decomposition: each FE KL = corresponding MI + “mismatch” KL(s) wrt baselines.
"""

import numpy as np

EPS = 1e-16  # numerical safety (nats)

# ---------- utils ----------
def normalize(v, axis=None):
    v = np.asarray(v, dtype=float)
    if axis is None:
        z = v.sum()
        return v / (z if z > 0 else 1.0)
    z = v.sum(axis=axis, keepdims=True)
    z = np.where(z <= 0, 1.0, z)
    return v / z

def xlogy(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((x > 0) & (y > 0), x * np.log(y), 0.0)

def kl_rowwise(p, q, axis=-1):
    """KL(p||q) computed rowwise along 'axis'; returns array with that axis removed."""
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p = p / np.clip(p.sum(axis=axis, keepdims=True), EPS, None)
    q = q / np.clip(q.sum(axis=axis, keepdims=True), EPS, None)
    return xlogy(p, (p + EPS) / (q + EPS)).sum(axis=axis)

def check_shapes(p_x, p_phi_x, p_s_xphi, r_phi, r_s, r_s_phi):
    X = p_x.shape[0]
    if p_phi_x.shape[0] != X: raise ValueError("p_phi_x first dim must match len(p_x)")
    P = p_phi_x.shape[1]
    if r_phi.shape[0] != P:   raise ValueError("r_phi length must match P")
    if p_s_xphi.shape[:2] != (X, P): raise ValueError("p_s_xphi first two dims must be (X,P)")
    S = p_s_xphi.shape[2]
    if r_s.shape[0] != S:     raise ValueError("r_s length must match S")
    if r_s_phi.shape != (P, S): raise ValueError("r_s_phi must be shape (P,S)")
    return X, P, S

# ---------- joint + marginals ----------
def build_joint(p_x, p_phi_x, p_s_xphi):
    p_x = np.asarray(p_x, dtype=float)
    p_phi_x = np.asarray(p_phi_x, dtype=float)
    p_s_xphi = np.asarray(p_s_xphi, dtype=float)
    if not np.isclose(p_x.sum(), 1.0, atol=1e-6): raise ValueError("p(x) must sum to 1.")
    if not np.allclose(p_phi_x.sum(axis=1), 1.0, atol=1e-6): raise ValueError("rows of p(φ|x) must sum to 1.")
    if not np.allclose(p_s_xphi.sum(axis=2), 1.0, atol=1e-6): raise ValueError("p(s|x,φ) rows must sum to 1.")
    joint = p_x[:, None, None] * p_phi_x[:, :, None] * p_s_xphi  # [X,P,S]
    Z = joint.sum()
    return joint / Z if not np.isclose(Z, 1.0, atol=1e-6) else joint

def induced_marginals(joint):
    # From joint [X,P,S]
    p_x     = joint.sum(axis=(1,2))     # [X]
    p_phi   = joint.sum(axis=(0,2))     # [P]
    p_s     = joint.sum(axis=(0,1))     # [S]
    p_x_phi = joint.sum(axis=2)         # [X,P]
    p_phi_s = joint.sum(axis=0)         # [P,S]
    p_s_phi = normalize(p_phi_s, axis=1)# [P,S]
    return dict(p_x=p_x, p_phi=p_phi, p_s=p_s, p_x_phi=p_x_phi, p_phi_s=p_phi_s, p_s_phi=p_s_phi)

# ---------- objectives ----------
def expected_utility(joint, U_xs):
    if U_xs is None: return 0.0
    X, P, S = joint.shape
    U = np.asarray(U_xs, dtype=float)
    if U.shape != (X, S): raise ValueError(f"U_xs must be [X,S]={X,S}, got {U.shape}")
    return (joint * U[:, None, :]).sum()

def mutual_infos(joint, mar, p_phi_x, p_s_xphi):
    # I(X;Φ) = Σ_{x,φ} p(x,φ) log p(φ|x)/p(φ)
    I_x_phi = xlogy(mar["p_x_phi"], (p_phi_x + EPS) / (mar["p_phi"][None,:] + EPS)).sum()
    # I(Φ;S) = Σ_{φ,s} p(φ,s) log p(s|φ)/p(s)
    I_phi_s = xlogy(mar["p_phi_s"], (mar["p_s_phi"] + EPS) / (mar["p_s"][None,:] + EPS)).sum()
    # I(X;S|Φ) = Σ_{x,φ,s} p(x,φ,s) log p(s|x,φ)/p(s|φ)
    ratio = (p_s_xphi + EPS) / (mar["p_s_phi"][None,:,:] + EPS)
    I_x_s_phi = xlogy(joint, ratio).sum()
    return I_x_phi, I_phi_s, I_x_s_phi

def gaussian_pdf_2d(x, mean, cov):
    # isotropic/diagonal covariance supported; cov is 2x2
    diff = x - mean
    det = np.linalg.det(cov)
    inv_xx = cov[0, 0]  # we only need σ for isotropic cov below
    inv_sigma = 1.0 / inv_xx
    norm_const = 1.0 / (2.0 * np.pi * np.sqrt(det))
    quad = -0.5 * inv_sigma * (diff[..., 0] ** 2 + diff[..., 1] ** 2)
    return norm_const * np.exp(quad)

def gaussian_likelihoods(xs, y_const, means, covariances):
    """
    Returns L[x_idx, s] = p(x|s) over a 1D sweep along x (y fixed).
    xs: [X], means: [S,2], covariances: [S,2,2]
    """
    Xs = np.stack([xs, np.full_like(xs, y_const)], axis=-1)  # [X,2]
    X = Xs.shape[0]
    S = means.shape[0]
    L = np.zeros((X, S), dtype=float)
    for s in range(S):
        L[:, s] = gaussian_pdf_2d(Xs, means[s], covariances[s])
    return L

def prepare_p_s_given_phi(M, p_s, is_mask=None, atol=1e-10):
    """
    If rows of M sum ≈ 1 (and M>=0), treat M as conditional p(s|φ).
    Otherwise treat M as mask/weights and set p(s|φ) ∝ M ⊙ p(s).
    Shapes: M [P,S], p_s [S].
    """
    M = np.asarray(M, dtype=float)
    P, S = M.shape
    row_sums = M.sum(axis=1)
    if is_mask is None:
        is_conditional = np.allclose(row_sums, 1.0, atol=atol) and np.all(M >= -atol)
    else:
        is_conditional = not is_mask
    if is_conditional:
        return normalize(M, axis=1)
    masked = M * p_s  # broadcast prior over states
    return normalize(masked, axis=1)

# ---------- convenience: build p(φ|x), p(s|x,φ) from mask M and Gaussians ----------
def build_conditionals_from_mask_and_gaussians(xs, y_const, means, covariances,
                                               M, p_s_prior=None, p_phi_prior=None):
    """
    Returns:
      p_x [X], p_phi_x [X,P], p_s_xphi [X,P,S], p_s_phi [P,S], L [X,S]
    where L[x,s] = p(x|s).
    """
    S = means.shape[0]
    if p_s_prior is None:
        p_s_prior = np.full(S, 1.0 / S)
    p_s_phi = prepare_p_s_given_phi(M, p_s_prior, is_mask=None)  # [P,S]
    P = p_s_phi.shape[0]

    # p(x): uniform over sampled xs
    X = len(xs)
    p_x = np.full(X, 1.0 / X)

    # Gaussian likelihoods
    L = gaussian_likelihoods(xs, y_const, means, covariances)  # [X,S]

    # Structure posterior p(φ|x) from evidence m_φ(x)=Σ_s p(x|s)p(s|φ)
    m = L @ p_s_phi.T                        # [X,P]
    if p_phi_prior is None:
        p_phi_prior = np.full(P, 1.0 / P)
    p_phi_x = normalize(p_phi_prior[None, :] * m, axis=1)  # [X,P]

    # State posterior per-structure p(s|x,φ) ∝ p(x|s)p(s|φ)
    p_s_xphi = normalize(p_s_phi[None, :, :] * L[:, None, :], axis=2)  # [X,P,S]

    return p_x, p_phi_x, p_s_xphi, p_s_phi, L

# --- add this helper once ---
def temper_structure_posterior(p_phi_x_base, alpha):
    """
    Rowwise tempering of p(φ|x):
      alpha = 1.0  -> identity
      alpha = 0.0  -> uniform over φ
      alpha = np.inf -> hard argmax per x
    """
    p = np.asarray(p_phi_x_base, dtype=float)
    if np.isinf(alpha):
        out = np.zeros_like(p)
        arg = np.argmax(p, axis=1)
        out[np.arange(p.shape[0]), arg] = 1.0
        return out
    return normalize((p + EPS) ** alpha, axis=1)


def free_energy_with_priors(p_x, p_phi_x, p_s_xphi, U_xs, betas,
                            r_phi, r_s, r_s_phi, alpha=1.0, gamma=1.0):
    """
    FE with *predefined* baselines:
      F = E[U] - (1/β1) E_x   KL(p(φ|x)   || r_phi)
                - (1/β2) E_φ   KL(p(s|φ)   || r_s)
                - (1/β3) E_{x,φ} KL(p(s|x,φ)|| r_s_phi[φ])

    alpha: temperature on p(φ|x)   (soft attention over structures)
    gamma: optional temperature on p(s|x,φ) (parallel path); keep 1.0 unless needed
    """
    beta1, beta2, beta3 = betas
    X, P, S = check_shapes(p_x, p_phi_x, p_s_xphi, r_phi, r_s, r_s_phi)

    # ---- apply temperatures (NEW) ----
    p_phi_x_eff = temper_structure_posterior(p_phi_x, alpha)
    if gamma == 1.0:
        p_s_xphi_eff = p_s_xphi
    else:
        p_s_xphi_eff = normalize((p_s_xphi + EPS) ** gamma, axis=2)

    # ---- build ONE joint with tempered conditionals ----
    joint = build_joint(p_x, p_phi_x_eff, p_s_xphi_eff)
    mar = induced_marginals(joint)

    # ---- E[U] ----
    EU = expected_utility(joint, U_xs)

    # ---- channel KLs vs predefined baselines (use tempered posteriors) ----
    # 1) E_x KL(p(φ|x) || r_phi)
    kl1_per_x = kl_rowwise(p_phi_x_eff, r_phi[None, :], axis=1)   # [X]
    KL1 = (mar["p_x"] * kl1_per_x).sum()

    # 2) E_φ KL(p(s|φ) || r_s) with induced p(s|φ) from the (tempered) joint
    kl2_per_phi = kl_rowwise(mar["p_s_phi"], r_s[None, :], axis=1)  # [P]
    KL2 = (mar["p_phi"] * kl2_per_phi).sum()

    # 3) E_{x,φ} KL(p(s|x,φ) || r_s|φ)
    r_s_phi_tiled = np.broadcast_to(r_s_phi[None, :, :], p_s_xphi_eff.shape)  # [X,P,S]
    kl3_per_xphi = kl_rowwise(p_s_xphi_eff, r_s_phi_tiled, axis=2)            # [X,P]
    KL3 = (mar["p_x_phi"] * kl3_per_xphi).sum()

    F = EU - (1.0/beta1)*KL1 - (1.0/beta2)*KL2 - (1.0/beta3)*KL3

    # ---- J_gen (optimal/self-consistent priors) from the SAME (tempered) joint ----
    I1, I2, I3 = mutual_infos(joint, mar, p_phi_x_eff, p_s_xphi_eff)
    J  = EU - (1.0/beta1)*I1 - (1.0/beta2)*I2 - (1.0/beta3)*I3
    I_eff = min(I1, I2) + I3   # optional: effective throughput

    # mismatch decompositions (gap between FE and J)
    mismatch_phi = xlogy(mar["p_phi"], (mar["p_phi"] + EPS)/(r_phi + EPS)).sum()
    mismatch_s   = xlogy(mar["p_s"],   (mar["p_s"]   + EPS)/(r_s   + EPS)).sum()
    mismatch_cond = (mar["p_phi"] * kl_rowwise(mar["p_s_phi"], r_s_phi, axis=1)).sum()

    return {
        "F_fixed": F,
        "J_gen": J,
        "E[U]": EU,
        "I(X;Phi)": I1, "I(Phi;S)": I2, "I(X;S|Phi)": I3,
        "I_eff(X;S)": I_eff,
        "KL_terms": {"E_x KL(phi|x||r_phi)": KL1,
                     "E_phi KL(s|phi||r_s)": KL2,
                     "E_xphi KL(s|x,phi||r_s|phi)": KL3},
        "mismatch": {"KL(p(phi)||r_phi)": mismatch_phi,
                     "KL(p(s)||r_s)": mismatch_s,
                     "E_phi KL(p(s|phi)||r_s|phi)": mismatch_cond},
        "joint": joint, "marginals": mar,
        "p_phi_x_tempered": p_phi_x_eff, "p_s_xphi_tempered": p_s_xphi_eff,
    }

# Shapes
# p_x: [X];            p_phi_x: [X,P];            p_s_xphi: [X,P,S]
# r_phi: [P];          r_s: [S];                  r_s_phi: [P,S]
# U_xs: [X,S] or None; betas = (beta1,beta2,beta3)

if __name__ == "__main__":
    # --- Gaussian config (3 states) ---
    config = [-96.0, 0.0, 96.0]
    sigma = 64.0 * 64.0
    means = np.array([[config[0], 0.0],
                      [config[1], 0.0],
                      [config[2], 0.0]], dtype=float)
    covariances = np.array([[[sigma, 0.0], [0.0, sigma]],
                            [[sigma, 0.0], [0.0, sigma]],
                            [[sigma, 0.0], [0.0, sigma]]], dtype=float)
    eps = 0.1
    xs = np.linspace(min(config) - eps, max(config) + eps, 201)
    y_const = 0.0

    # --- choose ANY mask M with shape [P,3]: P in {1,2,3} ---
    P = 1  # size of groups (structural models)

    if P == 1:
        M = np.array([[1, 1, 1]])                    # φ0: s1,s2,s3
    elif P == 2:
        M = np.array([[1, 1, 0], [0, 1, 1]])         # φ0: s1,s2; φ1: s2,s3
    elif P == 3:
        M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # singleton structures
    else:
        raise ValueError("Must define 0 < P ≤ S")
    
    
    # --- build conditionals and joint pieces from mask + Gaussians ---
    p_x, p_phi_x, p_s_xphi, p_s_phi, L = build_conditionals_from_mask_and_gaussians(
        xs, y_const, means, covariances, M,
        p_s_prior=None, p_phi_prior=None
    )

    # --- utility U(x,s). Replace with your task utility ---
    U_xs = np.log(L + EPS)

    #U_xs = -(xs[:, None] - means[:, 0][None, :]) ** 2 / (2 * (64.0 ** 2))

    # --- prices (inverse temperatures) for information costs ---
    beta1, beta2, beta3 = 1.0, 1.0, 1.0
    
    S = means.shape[0]
    r_phi = np.full(P, 1.0 / P)
    r_s = np.full(S, 1.0 / S)
    r_s_phi = M * r_s
    
    print("p(s) fixed", r_s)
    print("p(phi) fixed", r_phi)



    alpha = np.inf   # soften/harden structural attention (1.0 = off)
    gamma = 1.0   # (optional) temper p(s|x,φ)

    out = free_energy_with_priors(p_x, p_phi_x, p_s_xphi, U_xs,
                                betas=(1.0,1.0,1.0),
                                r_phi=r_phi, r_s=r_s, r_s_phi=r_s_phi,
                                alpha=alpha, gamma=gamma)

    print("F_fixed:", out["F_fixed"])
    print("J_gen :",  out["J_gen"])
    print("I_eff :",  out["I_eff(X;S)"])
    print(out["KL_terms"])
    print(out["mismatch"])