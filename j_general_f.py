# We'll create a new module `j_general_free_energy.py` that extends the user's script
# with a Free Energy F_gen computation using fixed priors (q_phi, q_s, q_s_given_phi),
# plus a small demonstration at the bottom (guarded by __main__).

import numpy as np
from typing import Optional, Dict, Any

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


def kl_cat(p, q):
    """KL(p || q) for categorical distributions along last axis."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return np.sum(xlogy(p, (p + EPS) / (q + EPS)), axis=-1)


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
        return normalize(M, axis=1)  # already conditional over s
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


# ---------- soft attention (structure temperature) ----------
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
    returns joint [X,P,S] with sum 1.
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


def induced_marginals(joint) -> Dict[str, np.ndarray]:
    """
    From joint [X,P,S] compute:
      p_x [X], p_phi [P], p_s [S], p_x_phi [X,P], p_phi_s [P,S],
      p_s_phi [P,S]=p(s|φ), p_x_given_phi [X,P]=p(x|φ).
    """
    p_x = joint.sum(axis=(1, 2))
    p_phi = joint.sum(axis=(0, 2))
    p_s = joint.sum(axis=(0, 1))
    p_x_phi = joint.sum(axis=2)       # [X,P]
    p_phi_s = joint.sum(axis=0)       # [P,S]
    p_s_phi = normalize(p_phi_s, axis=1)  # [P,S]
    p_x_given_phi = normalize(p_x_phi, axis=0)  # [X,P]
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

    I_x_phi = np.sum(xlogy(p_x_phi, (p_phi_x + EPS) / (p_phi[None, :] + EPS)))
    I_phi_s = np.sum(xlogy(p_phi_s, (p_s_phi + EPS) / (p_s[None, :] + EPS)))
    ratio = (p_s_xphi + EPS) / (p_s_phi[None, :, :] + EPS)
    I_x_s_phi = np.sum(xlogy(joint, ratio))
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
    return float(np.sum(joint * U_xs[:, None, :]))


# ---------- Free Energy with fixed priors ----------
def F_gen(
    p_x: np.ndarray,
    p_phi_x: np.ndarray,
    p_s_xphi: np.ndarray,
    U_xs: np.ndarray,
    beta1: float,
    beta2: float,
    beta3: float,
    q_phi: Optional[np.ndarray] = None,          # q(φ)
    q_s: Optional[np.ndarray] = None,            # q(s)
    q_s_given_phi: Optional[np.ndarray] = None,  # q(s|φ) [P,S]
    alpha: float = 1.0,
    return_parts: bool = False,
) -> Any:
    """
    Free energy with fixed priors:
      F = E[U]
          - (1/β1) E_{p(x)} KL( p(φ|x) || q(φ) )
          - (1/β2) E_{p(φ)} KL( p(s|φ) || q(s) )
          - (1/β3) E_{p(x,φ)} KL( p(s|x,φ) || q(s|φ) )

    Defaults (if q_* is None) choose the induced marginals/conditionals so
    that F == J_gen (no mismatch). Pass explicit q_* to include mismatch.
    """
    # Temper p(φ|x)
    p_phi_x_eff = temper_structure_posterior(p_phi_x, alpha)

    # Joint + marginals
    joint = build_joint(p_x, p_phi_x_eff, p_s_xphi)
    marg = induced_marginals(joint)

    # Defaults for q so that F==J when not provided
    P = p_phi_x.shape[1]
    S = p_s_xphi.shape[2]
    if q_phi is None:
        q_phi = marg["p_phi"].copy()
    if q_s is None:
        q_s = marg["p_s"].copy()
    if q_s_given_phi is None:
        q_s_given_phi = marg["p_s_phi"].copy()

    q_phi = normalize(q_phi)
    q_s = normalize(q_s)
    # Ensure q_s_given_phi rows sum to 1
    q_s_given_phi = normalize(q_s_given_phi, axis=1)

    # Expected KL terms under the current joint
    # 1) E_{p(x)} KL( p(φ|x) || q(φ) )
    KL1_avg = float(
        np.sum(marg["p_x"][:, None] * kl_cat(p_phi_x_eff, q_phi[None, :]))
    )

    # 2) E_{p(φ)} KL( p(s|φ) || q(s) )
    KL2_avg = float(
        np.sum(marg["p_phi"][:, None] * kl_cat(marg["p_s_phi"], q_s[None, :]))
    )

    # 3) E_{p(x,φ)} KL( p(s|x,φ) || q(s|φ) )
    KL3_point = kl_cat(p_s_xphi, q_s_given_phi[None, :, :])  # [X,P]
    KL3_avg = float(np.sum(marg["p_x_phi"] * KL3_point))

    # Free energy with fixed priors
    EU = expected_utility(joint, U_xs)
    F = EU - (1.0 / beta1) * KL1_avg - (1.0 / beta2) * KL2_avg - (1.0 / beta3) * KL3_avg

    if not return_parts:
        return F

    # Also compute J_gen and the mismatch decomposition for diagnostics
    I_x_phi, I_phi_s, I_x_s_phi = mutual_infos_from_joint(
        joint, marg, p_phi_x_eff, p_s_xphi
    )
    J = EU - (1.0 / beta1) * I_x_phi - (1.0 / beta2) * I_phi_s - (1.0 / beta3) * I_x_s_phi

    # Mismatch pieces for the identity:
    # E_x KL(p(φ|x)||q(φ)) = I(X;Φ) + KL(p(φ)||q(φ))
    mismatch1 = float(kl_cat(marg["p_phi"], q_phi))
    # E_φ KL(p(s|φ)||q(s)) = I(Φ;S) + KL(p(s)||q(s))
    mismatch2 = float(kl_cat(marg["p_s"], q_s))
    # E_{x,φ} KL(p(s|x,φ)||q(s|φ)) = I(X;S|Φ) + E_φ KL(p(s|φ)||q(s|φ))
    mismatch3 = float(np.sum(marg["p_phi"][:, None] * kl_cat(marg["p_s_phi"], q_s_given_phi)))

    # Sanity check identity (numerical)
    F_recon = J - (1.0 / beta1) * mismatch1 - (1.0 / beta2) * mismatch2 - (1.0 / beta3) * mismatch3
    identity_error = float(abs(F - F_recon))

    parts = {
        "F_gen": F,
        "E[U]": EU,
        "KL_terms": {
            "E_x KL(p(phi|x)||q(phi))": KL1_avg,
            "E_phi KL(p(s|phi)||q(s))": KL2_avg,
            "E_xphi KL(p(s|x,phi)||q(s|phi))": KL3_avg,
        },
        "J_gen": J,
        "MI_terms": {
            "I(X;Phi)": I_x_phi,
            "I(Phi;S)": I_phi_s,
            "I(X;S|Phi)": I_x_s_phi,
        },
        "Mismatch_decomp": {
            "KL(p(phi)||q(phi))": mismatch1,
            "KL(p(s)||q(s))": mismatch2,
            "E_phi KL(p(s|phi)||q(s|phi))": mismatch3,
            "Recon_error_|F - [J - mismatch]|": identity_error,
        },
        "marginals": marg,
        "joint": joint,
        "p_phi_x_tempered": p_phi_x_eff,
        "q_phi": q_phi, "q_s": q_s, "q_s_given_phi": q_s_given_phi,
    }
    return parts


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


# ---------- example usage ----------
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
    P = 1  # number of structures
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

    # --- prices (inverse temperatures) for information costs ---
    beta1, beta2, beta3 = 200, 100, 100

    # --- soft-attention temperature on structures ---
    alpha = np.inf  # 1.0 = Bayesian; 0.0 = uniform; np.inf = hard structure

    # ----- (A) Baseline: F with q's chosen as induced marginals => F == J -----
    parts_auto = F_gen(p_x, p_phi_x, p_s_xphi, U_xs, beta1, beta2, beta3,
                       q_phi=None, q_s=None, q_s_given_phi=None,
                       alpha=alpha, return_parts=True)
    print("\n=== Auto-q (q = induced) so F == J ===")
    print(f"J_gen: {parts_auto['J_gen']:.6f} | F_gen: {parts_auto['F_gen']:.6f}")
    print("Mismatch decomposition Recon error:", parts_auto["Mismatch_decomp"]["Recon_error_|F - [J - mismatch]|"])

    # ----- (B) Fixed priors: e.g., uniform q's to show mismatch terms -----
    q_phi = np.full(p_s_phi.shape[0], 1.0 / p_s_phi.shape[0])     # uniform over φ
    q_s = np.full(p_s_phi.shape[1], 1.0 / p_s_phi.shape[1])       # uniform over s
    #q_s_given_phi = np.tile(q_s[None, :], (p_s_phi.shape[0], 1))  # uniform s|φ for all φ
    if P == 2:
        q_s_given_phi = np.array([[1,0.5,0],
                                [0,0.5,1]])
    if P == 3:
        q_s_given_phi = np.array(np.eye(3))
    
    if P == 1:
        q_s_given_phi = np.tile(q_s[None, :], (p_s_phi.shape[0], 1))  # uniform s|φ for all φ
        

    print(q_s_given_phi)

    parts_fixed = F_gen(p_x, p_phi_x, p_s_xphi, U_xs, beta1, beta2, beta3,
                        q_phi=q_phi, q_s=q_s, q_s_given_phi=q_s_given_phi,
                        alpha=alpha, return_parts=True)
    print("\n=== Fixed-q (uniform) so F = J - weighted mismatch ===")
    print(f"J_gen: {parts_fixed['J_gen']:.6f} | F_gen: {parts_fixed['F_gen']:.6f}")
    print("KL_terms:", parts_fixed["KL_terms"])
    print("Mismatch_decomp:", parts_fixed["Mismatch_decomp"])
