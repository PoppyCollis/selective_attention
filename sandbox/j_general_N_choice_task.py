import numpy as np

EPS = 1e-16

# -------------------- utils --------------------
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

# -------------------- Gaussians --------------------
def gaussian_likelihoods_points(X_points, means, covariances):
    """
    X_points: [N,2]  evaluation points (each an (x,y))
    means:    [S,2]
    covs:     [S,2,2] (full or diagonal; positive-definite)
    returns:  L [N,S] with L[n,s] = N(Xn; mean_s, cov_s)
    """
    X_points = np.asarray(X_points, float)
    means = np.asarray(means, float)
    covariances = np.asarray(covariances, float)

    N = X_points.shape[0]
    S = means.shape[0]
    L = np.zeros((N, S), float)

    # precompute per-state inverses and log norms
    invs = []
    log_norms = []
    for s in range(S):
        Sigma = covariances[s]
        det = np.linalg.det(Sigma)
        inv = np.linalg.inv(Sigma)
        invs.append(inv)
        log_norms.append(-0.5 * (np.log((2.0 * np.pi) ** 2 * det)))
    invs = np.stack(invs, axis=0)          # [S,2,2]
    log_norms = np.array(log_norms)        # [S]

    # compute log pdf for each (n,s): -(1/2) (x-μ)^T Σ^{-1} (x-μ) + log_norm
    for s in range(S):
        diff = X_points - means[s]         # [N,2]
        quad = np.einsum('ni,ij,nj->n', diff, invs[s], diff)  # [N]
        logp = -0.5 * quad + log_norms[s]
        L[:, s] = np.exp(logp)

    return L  # [N,S]

# -------------------- structural priors --------------------
def coherent_structural_priors(M, p_s, leak=0.0):
    """
    From a mask M[P,S] and global prior p_s[S], build a *coherent* pair:
      p_phi[P], p_s_given_phi[P,S]
    s.t. sum_phi p_phi * p(s|phi) = p_s  (elementwise).
    """
    M = np.asarray(M, float)
    p_s = normalize(np.asarray(p_s, float))
    P, S = M.shape

    # (optional) leaky mask to avoid exact zeros if desired
    Mtil = leak + (1.0 - leak) * M
    cover = Mtil.sum(axis=0)                 # [S]
    if np.any(cover == 0):
        raise ValueError("Some states are not covered by any structure (zero column in M).")

    shares = Mtil * (p_s / cover)[None, :]   # [P,S]
    p_phi = shares.sum(axis=1)               # [P], sums to 1 automatically
    p_s_phi = shares / p_phi[:, None]        # [P,S] row-normalized
    return p_phi, p_s_phi

def temper_structure_posterior(p_phi_x, alpha):
    """
    Rowwise tempering of p(φ|x):
      alpha=1: identity; alpha=0: uniform; alpha=inf: hard argmax per x.
    """
    p = np.asarray(p_phi_x, float)
    if np.isinf(alpha):
        out = np.zeros_like(p)
        arg = np.argmax(p, axis=1)
        out[np.arange(p.shape[0]), arg] = 1.0
        return out
    return normalize((p + EPS) ** alpha, axis=1)

# -------------------- joint, MI, utility --------------------
def build_joint(p_x, p_phi_x, p_s_xphi):
    if not np.isclose(p_x.sum(), 1.0): raise ValueError("p(x) must sum to 1.")
    if not np.allclose(p_phi_x.sum(axis=1), 1.0): raise ValueError("rows of p(φ|x) must sum to 1.")
    if not np.allclose(p_s_xphi.sum(axis=2), 1.0): raise ValueError("p(s|x,φ) must sum to 1 over s.")
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

def expected_utility_loglik(joint, L):
    """
    Proper log-score utility: E[log p(x|s)] under the *same* joint.
    L: [N,S] likelihoods p(x|s) evaluated at the same X_points used for p(x).
    """
    if joint.shape[0] != L.shape[0]:
        raise ValueError("L and joint disagree on number of x points.")
    U_xs = np.log(L + EPS)     # [N,S]
    return (joint * U_xs[:, None, :]).sum()

# -------------------- build conditionals from (means,covs) + mask --------------------
def build_conditionals(means, covariances, X_points, M=None, p_s=None, leak=0.0):
    """
    means:        [S,2]
    covariances:  [S,2,2]
    X_points:     [N,2] (evaluation points; p(x) set uniform over them)
    M:            [P,S] binary/weight mask; if None -> identity (P=S)
    p_s:          [S] global prior over states; if None -> uniform
    leak:         ε leak in coherent priors (0.0 -> hard mask)

    returns:
      p_x [N], L [N,S], p_phi_prior [P], p_s_phi [P,S], p_phi_x [N,P], p_s_xphi [N,P,S]
    """
    means = np.asarray(means, float)
    covariances = np.asarray(covariances, float)
    S = means.shape[0]

    if p_s is None:
        p_s = np.full(S, 1.0 / S)

    if M is None:
        # default: singleton structures (no overlap)
        M = np.eye(S, dtype=float)
    else:
        M = np.asarray(M, float)
        if M.shape[1] != S:
            raise ValueError(f"M has {M.shape[1]} cols but S={S} states.")

    # Coherent p(φ), p(s|φ) so that Σφ p(φ)p(s|φ)=p(s)
    p_phi_prior, p_s_phi = coherent_structural_priors(M, p_s, leak=leak)  # [P], [P,S]
    P = p_s_phi.shape[0]

    # p(x) uniform over the given points
    N = X_points.shape[0]
    p_x = np.full(N, 1.0 / N)

    # Likelihoods p(x|s)
    L = gaussian_likelihoods_points(X_points, means, covariances)  # [N,S]

    # Structure posterior p(φ|x) ∝ p(φ) * m_φ(x), m_φ = Σ_s p(x|s)p(s|φ)
    m = L @ p_s_phi.T                          # [N,P]
    p_phi_x = normalize(p_phi_prior[None, :] * m, axis=1)  # [N,P]

    # Per-structure posteriors p(s|x,φ) ∝ p(x|s)p(s|φ)
    p_s_xphi = normalize(p_s_phi[None, :, :] * L[:, None, :], axis=2)  # [N,P,S]

    return p_x, L, p_phi_prior, p_s_phi, p_phi_x, p_s_xphi

# -------------------- top-level: run inference & J_gen --------------------
def run_inference(means, covariances, X_points, M=None, p_s=None,
                  alpha=1.0, gamma=1.0, betas=(1.0,1.0,1.0), leak=0.0,
                  return_all=False):
    """
    Flexible entry point. Size of |S| inferred from means/covariances.
    - If M is None -> P=S singleton structures; else supply any [P,S] mask.
    - alpha: temperature on p(φ|x) (soft attention over structures)
    - gamma: optional temperature on p(s|x,φ) (parallel path)
    - betas: (β1, β2, β3)
    - leak:  small ε in coherent priors to avoid hard zeros (e.g., 1e-6)

    Returns dict with J_gen, E[U], MI terms, effective throughput, and posteriors.
    """
    beta1, beta2, beta3 = betas

    p_x, L, p_phi_prior, p_s_phi, p_phi_x_base, p_s_xphi_base = build_conditionals(
        means, covariances, X_points, M=M, p_s=p_s, leak=leak
    )

    # Temper structure posterior and (optionally) within-structure state posteriors
    p_phi_x = temper_structure_posterior(p_phi_x_base, alpha)
    p_s_xphi = p_s_xphi_base if gamma == 1.0 else normalize((p_s_xphi_base + EPS) ** gamma, axis=2)

    joint = build_joint(p_x, p_phi_x, p_s_xphi)
    marg = induced_marginals(joint)

    # MI terms and effective throughput (nats)
    I1, I2, I3 = mutual_infos_from_joint(joint, marg, p_phi_x, p_s_xphi)
    I_eff = min(I1, I2) + I3

    # Expected utility (proper log score)
    EU = expected_utility_loglik(joint, L)

    # Generalized free energy
    J = EU - (1.0 / beta1) * I1 - (1.0 / beta2) * I2 - (1.0 / beta3) * I3

    out = dict(
        J_gen=J, E_U=EU,
        I_XPhi_nats=I1, I_PhiS_nats=I2, I_XS_given_Phi_nats=I3,
        I_eff_XS_nats=I_eff,
        p_x=p_x, L=L, p_phi_prior=p_phi_prior, p_s_phi=p_s_phi,
        p_phi_x=p_phi_x, p_s_xphi=p_s_xphi,
        marginals=marg, joint=joint
    )
    return out if return_all else {k: out[k] for k in ["J_gen","E_U","I_XPhi_nats","I_PhiS_nats","I_XS_given_Phi_nats","I_eff_XS_nats"]}

# -------------------- convenience to make a 1D x-line --------------------
def make_xline(means, n=401, y_const=0.0, pad=0.1):
    """
    Build a horizontal line of evaluation points spanning the min/max of the means (in x),
    slightly padded by 'pad'.
    """
    means = np.asarray(means, float)
    x_min = means[:,0].min() - pad
    x_max = means[:,0].max() + pad
    xs = np.linspace(x_min, x_max, n)
    X_points = np.stack([xs, np.full_like(xs, y_const)], axis=1)  # [n,2]
    return X_points, xs

# -------------------- example --------------------
if __name__ == "__main__":
    # Example 1: S=6 (six-alternative)
    S = 3
    #config = np.array([-240., -144., -48., 48., 144., 240.])  # any positions you like
    config = np.array([-96., 0., 96.])
    sigma = 64.0 * 64.0
    means = np.stack([config, np.zeros_like(config)], axis=1)  # [S,2]
    covs = np.array([[[sigma, 0.0], [0.0, sigma]]] * S)

    # Evaluation points along x
    X_points, xs = make_xline(means, n=401, y_const=0.0, pad=0.1)

    # Global prior over states (uniform here)
    p_s = np.full(S, 1.0 / S)

    # Mask M (optional). If None -> identity (P=S).
    # Example M: two wide overlapping groups across the 6 states.
    P = 1
    #M = np.zeros((2, S)); M[0, :S//2+1] = 1; M[1, S//2-1:] = 1
    #M = None  # identity by default (P=S singletons)
    if P == 1:
        M = np.array([[1, 1, 1]])                    # P=1  (φ0: s1,s2,s3)
    elif P == 2:
        M = np.array([[1, 1, 0], [0, 1, 1]]) # P=2  (φ0: s1,s2; φ1: s2,s3)
    elif P == 3:
        M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # P=3  (φ0: s1; φ1: s2; φ2: s3)
    else:
        raise ValueError("Must define 0 < P ≤ S")

    # Temperatures and prices
    alpha = 1.0   # soft attention over structures
    gamma = 1.0   # optional temperature on p(s|x,φ)
    betas = (1.0, 1.0, 1.0)

    out = run_inference(means, covs, X_points, M=M, p_s=p_s,
                        alpha=alpha, gamma=gamma, betas=betas, leak=0.0,
                        return_all=True)

    print(f"S={S}, P={'S' if M is None else M.shape[0]}, alpha={alpha}")
    print(f"J_gen               : {out['J_gen']:.6f}")
    print(f"E[U]                : {out['E_U']:.6f}")
    print(f"I(X;Phi) [nats]     : {out['I_XPhi_nats']:.6f}")
    print(f"I(Phi;S) [nats]     : {out['I_PhiS_nats']:.6f}")
    print(f"I(X;S|Phi) [nats]   : {out['I_XS_given_Phi_nats']:.6f}")
    print(f"I_eff(X;S) [nats]   : {out['I_eff_XS_nats']:.6f}")

    # Example 2: S=2 (two-alternative) – just pass 2 means & 2 covs
    means2 = np.array([[-48., 0.0], [48., 0.0]])
    covs2  = np.array([[[sigma, 0.0],[0.0, sigma]],
                       [[sigma, 0.0],[0.0, sigma]]])
    X_points2, _ = make_xline(means2, n=401, y_const=0.0, pad=0.1)
    out2 = run_inference(means2, covs2, X_points2, M=None, p_s=None,
                         alpha=1.0, gamma=1.0, betas=(1,1,1), leak=0.0,
                         return_all=False)
    print("\nTwo-choice task:")
    for k,v in out2.items():
        print(k, ":", f"{v:.6f}" if isinstance(v, float) else v)
