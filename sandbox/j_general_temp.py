# jgen_from_mask_gaussians.py  (with soft-attention α on p(φ|x))
# Build a single joint p(x,φ,s) from a gating mask M, Gaussian p(x|s),
# then compute J_gen with prices (β1, β2, β3). Actions a ≡ states s.

import numpy as np

EPS = 1e-16  # numerical safety, natural logs (nats)


# ---------- helpers ----------
def normalize(v, axis=None):
    v = np.asarray(v, dtype=float)
    if axis is None:
        z = v.sum()
        return v / (z if z > 0 else 1.0)
    z = v.sum(axis=axis, keepdims=True)
    z = np.where(z <= 0, 1.0, z)
    return v / z

def xlogy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((x > 0) & (y > 0), x * np.log(y), 0.0)

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
        return normalize(M, axis=1) # 1 is correct
    masked = M * p_s  # broadcast prior over states
    return normalize(masked, axis=1)

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


# ---------- soft attention (new) ----------
def temper_structure_posterior(p_phi_x_base, alpha):
    """
    Temper p(φ|x) rowwise:
      alpha=1: identity
      alpha=0: uniform over φ
      alpha=inf: hard argmax per x
    """
    p = np.asarray(p_phi_x_base, dtype=float)
    if np.isinf(alpha):
        out = np.zeros_like(p)
        arg = np.argmax(p, axis=1)
        out[np.arange(p.shape[0]), arg] = 1.0
        return out
    return normalize((p + EPS) ** alpha, axis=1)


# ---------- joint + marginals from p(x), p(φ|x), p(s|x,φ) ----------
def build_joint(p_x, p_phi_x, p_s_xphi):
    """
    p_x: [X]
    p_phi_x: [X,P]         == p(φ|x)
    p_s_xphi: [X,P,S]      == p(s|x,φ)
    returns joint [X,P,S]
    """
    p_x = np.asarray(p_x, dtype=float)
    p_phi_x = np.asarray(p_phi_x, dtype=float)
    p_s_xphi = np.asarray(p_s_xphi, dtype=float)
    if not np.isclose(p_x.sum(), 1.0, atol=1e-6):
        raise ValueError("p(x) must sum to 1.")
    if not np.allclose(p_phi_x.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Rows of p(φ|x) must sum to 1.")
    if not np.allclose(p_s_xphi.sum(axis=2), 1.0, atol=1e-6):
        raise ValueError("For each (x,φ), p(s|x,φ) must sum to 1 over s.")
    joint = p_x[:, None, None] * p_phi_x[:, :, None] * p_s_xphi
    Z = joint.sum()
    if not np.isclose(Z, 1.0, atol=1e-6):
        joint = joint / Z
    return joint

def induced_marginals(joint):
    """
    From joint [X,P,S] compute:
      p_x [X], p_phi [P], p_s [S], p_x_phi [X,P], p_phi_s [P,S],
      p_s_phi [P,S]=p(s|φ), p_x_given_phi [X,P]=p(x|φ).
    """
    p_x = joint.sum(axis=(1, 2))
    p_phi = joint.sum(axis=(0, 2))
    p_s = joint.sum(axis=(0, 1))
    p_x_phi = joint.sum(axis=2)
    p_phi_s = joint.sum(axis=0)
    p_s_phi = normalize(p_phi_s, axis=1)
    p_x_given_phi = normalize(p_x_phi, axis=0)
    return dict(p_x=p_x, p_phi=p_phi, p_s=p_s,
                p_x_phi=p_x_phi, p_phi_s=p_phi_s,
                p_s_phi=p_s_phi, p_x_given_phi=p_x_given_phi)

def mutual_infos_from_joint(joint, marginals, p_phi_x, p_s_xphi):
    """
    I(X;Φ)          = Σ_{x,φ} p(x,φ) log [ p(φ|x) / p(φ) ]
    I(Φ;S)          = Σ_{φ,s} p(φ,s) log [ p(s|φ) / p(s) ]
    I(X;S|Φ)        = Σ_{x,φ,s} p(x,φ,s) log [ p(s|x,φ) / p(s|φ) ]
    """
    p_x_phi = marginals["p_x_phi"]   # [X,P]
    p_phi   = marginals["p_phi"]     # [P]
    p_phi_s = marginals["p_phi_s"]   # [P,S]
    p_s     = marginals["p_s"]       # [S]
    p_s_phi = marginals["p_s_phi"]   # [P,S]
    
    print("p_s, p_phi, p_s_phi", p_s, p_phi, p_s_phi)

    I_x_phi = xlogy(p_x_phi, (p_phi_x + EPS) / (p_phi[None, :] + EPS)).sum()
    I_phi_s = xlogy(p_phi_s, (p_s_phi + EPS) / (p_s[None, :] + EPS)).sum()
    ratio = (p_s_xphi + EPS) / (p_s_phi[None, :, :] + EPS)
    I_x_s_phi = xlogy(joint, ratio).sum()
    return I_x_phi, I_phi_s, I_x_s_phi

def expected_utility(joint, U_xs):
    """
    E[U] = Σ_{x,φ,s} p(x,φ,s) U(x,s)
    U_xs: [X,S]
    """
    X, P, S = joint.shape
    U_xs = np.asarray(U_xs, dtype=float)
    if U_xs.shape != (X, S):
        raise ValueError(f"U_xs must be [X,S]={X,S}, got {U_xs.shape}")
    return (joint * U_xs[:, None, :]).sum()


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
       p_s_prior = np.full(S, 1.0 / S) #p_s_prior = np.array([0.30, 0.40, 0.30])
    p_s_phi = prepare_p_s_given_phi(M, p_s_prior, is_mask=None)  # [P,S]
    P = p_s_phi.shape[0]
    print(p_s_phi)

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


# ---------- top-level: J_gen with α (and optional γ) ----------
def J_gen(p_x, p_phi_x, p_s_xphi, U_xs, beta1, beta2, beta3,
          alpha=1.0, return_parts=False):
    """
    J_gen = E[U] - (1/β1) I(X;Φ) - (1/β2) I(Φ;S) - (1/β3) I(X;S|Φ),
    computed from ONE joint p(x,φ,s)=p(x)·temper(p(φ|x),α)·temper(p(s|x,φ),γ).

    alpha: temperature on p(φ|x)   (soft attention over structures)
    """
    # Temper p(φ|x)
    p_phi_x_eff = temper_structure_posterior(p_phi_x, alpha)

    # Optional: temper p(s|x,φ) row-wise over s

    # Build joint with tempered conditionals
    joint = build_joint(p_x, p_phi_x_eff, p_s_xphi)
    marg = induced_marginals(joint)

    # MI and EU from the same joint (use tempered conditionals in MI)
    I_x_phi, I_phi_s, I_x_s_phi = mutual_infos_from_joint(joint, marg, p_phi_x_eff, p_s_xphi)
    I_x_s = min(((1.0 / beta1)*I_x_phi),((1.0 / beta2)* I_phi_s)) + ((1.0 / beta3)*I_x_s_phi)
    EU = expected_utility(joint, U_xs) 
    J_eff = EU - I_x_s

    J = EU - (1.0 / beta1) * I_x_phi - (1.0 / beta2) * I_phi_s - (1.0 / beta3) * I_x_s_phi
    if not return_parts:
        return J
    return {
        "J_gen": J, "E[U]": EU,
        "I(X;Phi)": (1.0 / beta1) *I_x_phi, "I(Phi;S)": (1.0 / beta2) *I_phi_s, "I(X;S|Phi)": (1.0 / beta3) *I_x_s_phi, "I(X;S)": I_x_s,
        "joint": joint, "marginals": marg,
        "p_phi_x_tempered": p_phi_x_eff,
        "p_s_xphi": p_s_xphi,
        "Effective J": J_eff
    }


# ---------- example usage (mirrors your Gaussians + flexible mask) ----------
if __name__ == "__main__":
    # --- Gaussian config (3 states) ---
    config = [-1.0, 0.0, 1.0]
    sigma = 1
    means = np.array([[config[0], 0.0],
                      [config[1], 0.0],
                      [config[2], 0.0]], dtype=float)
    covariances = np.array([[[sigma, 0.0], [0.0, sigma]],
                            [[sigma, 0.0], [0.0, sigma]],
                            [[sigma, 0.0], [0.0, sigma]]], dtype=float)
    eps = 0.5
    xs = np.linspace(min(config) - eps, max(config) + eps, 10000)
    y_const = 0.0

    # --- choose ANY mask M with shape [P,3]: P in {1,2,3} ---
    P = 2  # size of groups (structural models)

    if P == 1:
        M = np.array([[1, 1, 1]])                    # φ0: s1,s2,s3
    elif P == 2:
        #M = np.array([[1, 1, 0], [0, 1, 1]])
        M = np.array([[0.4285, 0.5714, 0], [0, 0.57142,  0.428571]])       # φ0: s1,s2; φ1: s2,s3
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

    # --- prices (inverse temperatures) for information costs --- should be 1 over?
    beta1, beta2, beta3 = np.inf,2.5,2

    # --- soft-attention temperature on structures ---
    alpha = np.inf # 1.0 = Bayesian; 0.0 = uniform; np.inf = hard structure

    parts = J_gen(p_x, p_phi_x, p_s_xphi, U_xs, beta1, beta2, beta3,
                  alpha=alpha, return_parts=True)

    print(f"J_gen         : {parts['J_gen']:.26f}")
    print(f"Effective J : {parts['Effective J']:.26f}")
    print(f"E[U]          : {parts['E[U]']:.26f}")
    print(f"I(X;Phi)      : {parts['I(X;Phi)']:.26f}")
    print(f"I(Phi;S)      : {parts['I(Phi;S)']:.26f}")
    print(f"I(X;S|Phi)    : {parts['I(X;S|Phi)']:.26f}")
    print(f"I(X;S) Effective throughput : {parts['I(X;S)']:.26f}")
