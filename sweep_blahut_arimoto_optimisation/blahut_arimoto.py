import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any, Union
from dataclasses import dataclass
import pandas as pd

EPS = 1e-16

def _argmax_dist(scores: np.ndarray, prior: Optional[np.ndarray] = None) -> np.ndarray:
    scores = np.asarray(scores, float)
    winners = (scores == np.max(scores))
    out = np.zeros_like(scores, float)
    if prior is None:
        out[winners] = 1.0 / (np.sum(winners) + EPS)
    else:
        w = np.clip(np.asarray(prior, float)[winners], EPS, np.inf)
        out[winners] = w / (np.sum(w) + EPS)
    return out

def safe_softmax(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    # If any non-finite appears, fall back to argmax along `axis`
    if not np.all(np.isfinite(logits)):
        # argmax along axis, uniform over ties
        m = np.nanmax(logits, axis=axis, keepdims=True)
        winners = (logits == m).astype(float)
        return winners / (np.sum(winners, axis=axis, keepdims=True) + EPS)
    m = np.max(logits, axis=axis, keepdims=True)
    z = np.exp(logits - m)
    return z / (np.sum(z, axis=axis, keepdims=True) + EPS)

def boltzmann_dist(prior: np.ndarray, beta: float, utility: np.ndarray) -> np.ndarray:
    prior = np.clip(prior, EPS, 1.0)
    if beta is None or np.isinf(beta):
        # β→∞ ⇒ greedy: argmax on utility (ties broken by prior mass)
        return _argmax_dist(utility, prior=prior)
    logits = np.log(prior) + beta * utility
    return safe_softmax(logits, axis=0)


def _normalize(v: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    v = np.clip(v, 0.0, np.inf)
    s = np.sum(v, axis=axis, keepdims=True) + EPS
    return v / s

def softmax(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    m = np.max(logits, axis=axis, keepdims=True)
    z = np.exp(logits - m)
    return z / (np.sum(z, axis=axis, keepdims=True) + EPS)

def log_bits(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1.0))

def kl_bits(p: np.ndarray, q: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)
    val = np.sum(p * (np.log(p) - np.log(q)), axis=axis)
    return np.maximum(val, 0.0)


def marginalizeo(pogw: np.ndarray, pagow: np.ndarray) -> np.ndarray:
    A, X, N = pagow.shape
    assert pogw.shape == (X, N)
    pagw = np.empty((A, N))
    for j in range(N):
        pagw[:, j] = pagow[:, :, j] @ pogw[:, j]**(1)
    return _normalize(pagw, axis=0)

def compute_marginals(pw: np.ndarray, pogw: np.ndarray, pagow: np.ndarray):
    X, N = pogw.shape
    A = pagow.shape[0]
    assert pagow.shape == (A, X, N)
    po = pogw @ pw
    po = _normalize(po)
    pagw = marginalizeo(pogw, pagow)
    pa = pagw @ pw
    pa = _normalize(pa)
    return po, pa, pagw

def compute_pago_iteration(pogw: np.ndarray, pagow: np.ndarray, beta2: float, beta3: float,
                           U_pre: np.ndarray, pa: np.ndarray, po: np.ndarray, pw: np.ndarray) -> np.ndarray:
    A, X, N = pagow.shape
    assert pogw.shape == (X, N)
    assert U_pre.shape == (A, N)
    assert pa.shape == (A,)
    assert po.shape == (X,)
    assert pw.shape == (N,)
    pago = np.zeros((A, X))
    for x in range(X):
        w_post = pogw[x, :] * pw
        w_post = w_post / (np.sum(w_post) + EPS)
        if beta3 == 0 or np.isinf(beta3):
            EU = U_pre @ w_post
            pago[:, x] = boltzmann_dist(pa, beta2, EU)
        else:
            avg = np.zeros(A)
            for j in range(N):
                avg += w_post[j] * pagow[:, x, j]
            pago[:, x] = _normalize(avg)
    return pago

def compute_pagow_iteration(pago: np.ndarray, beta2: float, beta3: float,
                            U_pre: np.ndarray, pa: np.ndarray) -> np.ndarray:
    A, X = pago.shape
    N = U_pre.shape[1]
    assert U_pre.shape == (A, N)
    assert pa.shape == (A,)
    pagow = np.empty((A, X, N))
    inv_beta2 = 0.0 if (beta2 is None or np.isinf(beta2)) else (1.0 / beta2)
    for j in range(N):
        for x in range(X):
            if (beta3 == 0) or np.isinf(beta3):
                pagow[:, x, j] = pago[:, x]
            else:
                corr = - (beta3 * inv_beta2) * (np.log(np.clip(pago[:, x] + EPS, EPS, 1.0)) - np.log(np.clip(pa + EPS, EPS, 1.0)))
                logits = np.log(np.clip(pago[:, x], EPS, 1.0)) + beta3 * U_pre[:, j] + corr
                pagow[:, x, j] = softmax(logits, axis=0)
    return pagow

def compute_pogw_iteration(beta1, beta2, beta3, U_pre, pa, po, pago, pagow):
    A, X = pago.shape
    N = U_pre.shape[1]
    inv_beta2 = 0.0 if (beta2 is None or np.isinf(beta2)) else (1.0 / beta2)
    inv_beta3 = 0.0 if (beta3 is None or np.isinf(beta3)) else (1.0 / beta3)
    pogw = np.empty((X, N))
    for j in range(N):
        util_x = np.empty(X)
        for x in range(X):
            p_axw = np.clip(pagow[:, x, j], EPS, 1.0)
            EU = float(np.dot(p_axw, U_pre[:, j]))
            DKL_a   = float(np.sum(p_axw * (np.log(p_axw) - np.log(np.clip(pa,   EPS, 1.0)))))
            DKL_ago = float(np.sum(p_axw * (np.log(p_axw) - np.log(np.clip(pago[:, x], EPS, 1.0)))))
            util_x[x] = EU - inv_beta2 * DKL_a - (inv_beta3 - inv_beta2) * DKL_ago
        if beta1 is None or np.isinf(beta1):
            pogw[:, j] = _argmax_dist(util_x)  # ignore po in the β→∞ limit
        else:
            logits_x = np.log(np.clip(po, EPS, 1.0)) + beta1 * util_x
            pogw[:, j] = safe_softmax(logits_x, axis=0)
    return pogw


@dataclass
class BAResult:
    po: np.ndarray
    pa: np.ndarray
    pogw: np.ndarray
    pago: np.ndarray
    pagow: np.ndarray
    pagw: np.ndarray
    history: Optional[Dict[str, Any]] = None

def threevar_BA_iterations(X: int, beta1: float, beta2: float, beta3: float,
                           U_pre: np.ndarray, pw: np.ndarray,
                           tol: float = 1e-10, maxiter: int = 10000,
                           init_pogw_uniformly: bool = False,
                           init_pogw_sparse: bool = True,
                           init_pagow_uniformly: bool = True,
                           track_history: bool = False) -> BAResult:
    A, N = U_pre.shape
    assert pw.shape == (N,)
    assert np.all(pw >= 0) and abs(np.sum(pw) - 1.0) < 1e-6
    if init_pogw_uniformly:
        pogw = np.full((X, N), 1.0 / X)
    elif init_pogw_sparse:
        pogw = np.zeros((X, N))
        ks = np.random.randint(0, X, size=N)
        pogw[ks, np.arange(N)] = 1.0
    else:
        pogw = _normalize(np.random.rand(X, N), axis=0)
    if init_pagow_uniformly:
        pago = np.full((A, X), 1.0 / A)
        pagow = np.repeat(pago[:, :, None], N, axis=2)
    else:
        pago = _normalize(np.random.rand(A, X), axis=0)
        pagow = np.repeat(pago[:, :, None], N, axis=2)
    history = {"po": [], "pa": [], "pogw": [], "pago": [], "pagow": []} if track_history else None
    prev_pogw = pogw.copy()
    prev_pagow = pagow.copy()
    for it in range(1, maxiter + 1):
        po, pa, pagw = compute_marginals(pw, pogw, pagow)
        pago = compute_pago_iteration(pogw, pagow, beta2, beta3, U_pre, pa, po, pw)
        pagow = compute_pagow_iteration(pago, beta2, beta3, U_pre, pa)
        pogw = compute_pogw_iteration(beta1, beta2, beta3, U_pre, pa, po, pago, pagow)
        delta = max(np.max(np.abs(pogw - prev_pogw)), np.max(np.abs(pagow - prev_pagow)))
        prev_pogw[...] = pogw
        prev_pagow[...] = pagow
        if history is not None:
            history["po"].append(po.copy())
            history["pa"].append(pa.copy())
            history["pogw"].append(pogw.copy())
            history["pago"].append(pago.copy())
            history["pagow"].append(pagow.copy())
        if delta < tol:
            break
    return BAResult(po=po, pa=pa, pogw=pogw, pago=pago, pagow=pagow, pagw=pagw, history=history)


def threevar_BA_iterations_warm(
    X: int, beta1: float, beta2: float, beta3: float,
    U_pre: np.ndarray, pw: np.ndarray,
    tol: float = 1e-10, maxiter: int = 10000,
    # NEW: optional initializations
    init_pogw: Optional[np.ndarray] = None,
    init_pagow: Optional[np.ndarray] = None,
    # Fallbacks if inits are None
    init_pogw_uniformly: bool = False,
    init_pogw_sparse: bool = True,
    init_pagow_uniformly: bool = True,
    # NEW: light perturbation to avoid path lock-in
    perturb_std: float = 0.0,
    track_history: bool = False
) -> BAResult:
    """
    Identical fixed-point map as your threevar_BA_iterations, but accepts optional
    warm-start arrays for pogw (X,N) and pagow (A,X,N). If provided, those take precedence.
    Optionally adds small Gaussian noise before normalizing (perturb_std).
    """
    A, N = U_pre.shape
    assert pw.shape == (N,)
    assert np.all(pw >= 0) and abs(np.sum(pw) - 1.0) < 1e-6

    # Initialize pogw
    if init_pogw is not None:
        pogw = np.array(init_pogw, float, copy=True)
        assert pogw.shape == (X, N)
        if perturb_std > 0:
            pogw = np.maximum(pogw + perturb_std * np.random.randn(X, N), 0.0)
        pogw = _normalize(pogw, axis=0)
    elif init_pogw_uniformly:
        pogw = np.full((X, N), 1.0 / X)
    elif init_pogw_sparse:
        pogw = np.zeros((X, N))
        ks = np.random.randint(0, X, size=N)
        pogw[ks, np.arange(N)] = 1.0
    else:
        pogw = _normalize(np.random.rand(X, N), axis=0)

    # Initialize pagow (and pago implied by it)
    if init_pagow is not None:
        pagow = np.array(init_pagow, float, copy=True)
        assert pagow.shape == (A, X, N)
        if perturb_std > 0:
            pagow = np.maximum(pagow + perturb_std * np.random.randn(A, X, N), 0.0)
        # normalize each (x,w) column over a
        s = np.sum(pagow, axis=0, keepdims=True) + EPS
        pagow = pagow / s
    elif init_pagow_uniformly:
        pago = np.full((A, X), 1.0 / A)
        pagow = np.repeat(pago[:, :, None], N, axis=2)
    else:
        pago = _normalize(np.random.rand(A, X), axis=0)
        pagow = np.repeat(pago[:, :, None], N, axis=2)

    history = {"po": [], "pa": [], "pogw": [], "pago": [], "pagow": []} if track_history else None

    prev_pogw = pogw.copy()
    prev_pagow = pagow.copy()

    for it in range(1, maxiter + 1):
        po, pa, pagw = compute_marginals(pw, pogw, pagow)
        pago = compute_pago_iteration(pogw, pagow, beta2, beta3, U_pre, pa, po, pw)
        pagow = compute_pagow_iteration(pago, beta2, beta3, U_pre, pa)
        pogw = compute_pogw_iteration(beta1, beta2, beta3, U_pre, pa, po, pago, pagow)
        delta = max(np.max(np.abs(pogw - prev_pogw)), np.max(np.abs(pagow - prev_pagow)))
        prev_pogw[...] = pogw
        prev_pagow[...] = pagow
        if history is not None:
            history["po"].append(po.copy())
            history["pa"].append(pa.copy())
            history["pogw"].append(pogw.copy())
            history["pago"].append(pago.copy())
            history["pagow"].append(pagow.copy())
        if delta < tol:
            break

    return BAResult(po=po, pa=pa, pogw=pogw, pago=pago, pagow=pagow, pagw=pagw, history=history)

def make_w_samples(L: float, Uhi: float, N: int, grid: bool = False, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    if not (Uhi > L):
        raise ValueError("need U > L")
    if rng is None:
        rng = np.random.default_rng()
    if grid:
        w = np.linspace(L, Uhi, N)
        if N == 1:
            s = np.array([1.0])
        else:
            Δ = (Uhi - L) / (N - 1)
            s = np.full(N, Δ / (Uhi - L))
            s[0] *= 0.5
            s[-1] *= 0.5
            s /= np.sum(s)
    else:
        w = rng.uniform(L, Uhi, size=N)
        s = np.full(N, 1.0 / N)
    return w.astype(float), s.astype(float)

def U_fn(a: int, w: float, mu_as, sigma_as) -> float: 
    mu = mu_as[a-1] 
    sigma = sigma_as[a-1] 
    return -0.5*np.log(2*np.pi*sigma**2) - 0.5*((w - mu)**2)/(sigma**2)

# suppose you have log_p_a_given_x: shape (A, X)
def build_U_pre(U_fn, A, w, mu_as, sigma_as):
    N = len(w)
    U = np.empty((A, N))
    for j in range(N):
        for a in range(A):
            U[a, j] = float(U_fn(a + 1, float(w[j]), mu_as, sigma_as))
    return U


def threevar_BA_continuousW(X: int, beta1: float, beta2: float, beta3: float,
                            U_fn: Callable[[int, float], float], A: int,
                            L: float, Uhi: float, N: int, tol: float = 1e-10,
                            maxiter: int = 10000, grid: bool = False, **init_opts: Any) -> BAResult:
    # w, pw = make_w_samples_gaussian(L, Uhi, N, grid=grid)
    w, pw = make_w_samples(L, Uhi, N, grid=grid)

    U_pre = build_U_pre(U_fn, A, w)
    return threevar_BA_iterations(X=X, beta1=beta1, beta2=beta2, beta3=beta3,
                                  U_pre=U_pre, pw=pw, tol=tol, maxiter=maxiter, **init_opts)

def entropy(p, axis=None):
    p = np.clip(p, EPS, 1.0)
    return -np.sum(p * np.log(p), axis=axis)

def mutual_info_XW(pogw, po, pw):
    # I(X;W) = E_w[ KL(p(x|w) || p(x)) ]
    X, N = pogw.shape
    total = 0.0
    for j in range(N):
        px_w = np.clip(pogw[:, j], EPS, 1.0)
        total += pw[j] * np.sum(px_w * (np.log(px_w) - np.log(np.clip(po, EPS, 1.0))))
    return float(total)

def mutual_info_AO(pago, pa, po):
    # I(A;O) = sum_x p(x) KL(p(a|x) || p(a))
    A, X = pago.shape
    total = 0.0
    for x in range(X):
        pax = np.clip(pago[:, x], EPS, 1.0)
        total += po[x] * np.sum(pax * (np.log(pax) - np.log(np.clip(pa, EPS, 1.0))))
    return float(total)

def mutual_info_WA_given_O(pagow, pago, pogw, po, pw):
    # I(W;A|O) = sum_x p(x) E_{w|x}[ KL(p(a|x,w) || p(a|x)) ]
    A, X, N = pagow.shape
    total = 0.0
    for x in range(X):
        # p(w|x) over samples
        w_post = pogw[x, :] * pw
        denom = float(np.sum(w_post) + EPS)
        if denom <= 0:
            continue
        w_post /= denom
        pax = np.clip(pago[:, x], EPS, 1.0)
        for j in range(N):
            p_axw = np.clip(pagow[:, x, j], EPS, 1.0)
            total += po[x] * w_post[j] * np.sum(p_axw * (np.log(p_axw) - np.log(pax)))
    return float(total)

def mutual_info_AW(pagw, pa, pw):
    # I(A;W) = E_w[ KL(p(a|w) || p(a)) ]
    A, N = pagw.shape
    total = 0.0
    for j in range(N):
        paw = np.clip(pagw[:, j], EPS, 1.0)
        total += pw[j] * np.sum(paw * (np.log(paw) - np.log(np.clip(pa, EPS, 1.0))))
    return float(total)

def expected_utility(U_pre, pogw, pagow, pw):
    # E[U] = E_w E_{x|w} E_{a|x,w} [ U(a,w) ]
    A, N = U_pre.shape
    X, N2 = pogw.shape
    assert N == N2
    total = 0.0
    for j in range(N):
        for x in range(X):
            total += pw[j] * pogw[x, j] * float(np.dot(pagow[:, x, j], U_pre[:, j]))
    return float(total)

def objective_value(EU, I_ow, I_ao, I_wago, beta1, beta2, beta3):
    inv_b1 = 0.0 if (beta1 is None or np.isinf(beta1)) else 1.0 / beta1
    inv_b2 = 0.0 if (beta2 is None or np.isinf(beta2)) else 1.0 / beta2
    inv_b3 = 0.0 if (beta3 is None or np.isinf(beta3)) else 1.0 / beta3
    # General/parallel composite (matches terms in update equations):
    # J = EU - ((1/β1)* I(X;W)) - ((1/β2) *I(A;W)) - ((1/β3) * I(A;W|X))
    return EU - (inv_b1 * I_ow) - (inv_b2 * I_ao) - (inv_b3 * I_wago)

def objective_from_state(U_pre, pw, beta1, beta2, beta3, po, pa, pogw, pago, pagow) -> float:
    pagw = marginalizeo(pogw, pagow)
    I_ow = mutual_info_XW(pogw, po, pw)
    I_ao = mutual_info_AO(pago, pa, po)
    I_wago = mutual_info_WA_given_O(pagow, pago, pogw, po, pw)
    EU = expected_utility(U_pre, pogw, pagow, pw)
    return objective_value(EU, I_ow, I_ao, I_wago, beta1, beta2, beta3)

def collect_metrics_over_history(history, pw, U_pre, beta1, beta2, beta3):
    rows = []
    for po, pa, pogw, pago, pagow in zip(history["po"], history["pa"], history["pogw"], history["pago"], history["pagow"]):
        pagw = marginalizeo(pogw, pagow)
        I_ow = mutual_info_XW(pogw, po, pw)
        I_ao = mutual_info_AO(pago, pa, po)
        I_wago = mutual_info_WA_given_O(pagow, pago, pogw, po, pw)
        I_aw = mutual_info_AW(pagw, pa, pw)
        EU = expected_utility(U_pre, pogw, pagow, pw)
        J = objective_value(EU, I_ow, I_ao, I_wago, beta1, beta2, beta3)
        rows.append(dict(I_ow=I_ow, I_ao=I_ao, I_wago=I_wago, I_aw=I_aw, E_U=EU, Objective_value=J))
    df = pd.DataFrame(rows)
    df.index.name = "iteration"
    return df
    