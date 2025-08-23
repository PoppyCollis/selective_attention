# parallel Blahut–Arimoto notebook

import numpy as np
np.set_printoptions(precision=6, suppress=True)
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

EPS = 1e-16

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

def boltzmann_dist(prior: np.ndarray, beta: float, utility: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(prior, EPS, 1.0)) + beta * utility
    return softmax(logits, axis=0)

def marginalizeo(pogw: np.ndarray, pagow: np.ndarray) -> np.ndarray:
    A, X, N = pagow.shape
    assert pogw.shape == (X, N)
    pagw = np.empty((A, N))
    for j in range(N):
        pagw[:, j] = pagow[:, :, j] @ pogw[:, j]
    return _normalize(pagw, axis=0)

def compute_marginals(pw: np.ndarray, pogw: np.ndarray, pagow: np.ndarray):
    X, N = pogw.shape
    A = pagow.shape[0]
    assert pagow.shape == (A, X, N)
    po = pogw @ pw
    po = _normalize(po)
    pagw = marginalizeo(pogw, pagow) # changed pzc
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

def compute_pogw_iteration(beta1: float, beta2: float, beta3: float,
                           U_pre: np.ndarray, pa: np.ndarray, po: np.ndarray,
                           pago: np.ndarray, pagow: np.ndarray) -> np.ndarray:
    A, X = pago.shape
    N = U_pre.shape[1]
    assert pagow.shape == (A, X, N)
    inv_beta2 = 0.0 if (beta2 is None or np.isinf(beta2)) else (1.0 / beta2)
    inv_beta3 = 0.0 if (beta3 is None or np.isinf(beta3)) else (1.0 / beta3)
    pogw = np.empty((X, N))
    for j in range(N):
        logits_x = np.empty(X)
        for x in range(X):
            p_axw = pagow[:, x, j]
            EU = float(np.dot(p_axw, U_pre[:, j]))
            DKL_a = float(np.sum(p_axw * (np.log(np.clip(p_axw, EPS, 1.0)) - np.log(np.clip(pa, EPS, 1.0)))))
            DKL_ago = float(np.sum(p_axw * (np.log(np.clip(p_axw, EPS, 1.0)) - np.log(np.clip(pago[:, x], EPS, 1.0)))))
            util = EU - inv_beta2 * DKL_a - (inv_beta3 - inv_beta2) * DKL_ago
            logits_x[x] = np.log(np.clip(po[x], EPS, 1.0)) + beta1 * util
        pogw[:, j] = softmax(logits_x, axis=0)
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

def build_U_pre(U_fn: Callable[[int, float], float], A: int, w: np.ndarray) -> np.ndarray:
    N = len(w)
    U_pre = np.empty((A, N), dtype=float)
    for j in range(N):
        for a in range(A):
            U_pre[a, j] = float(U_fn(a + 1, float(w[j])))
    return U_pre

def threevar_BA_continuousW(X: int, beta1: float, beta2: float, beta3: float,
                            U_fn: Callable[[int, float], float], A: int,
                            L: float, Uhi: float, N: int, tol: float = 1e-10,
                            maxiter: int = 10000, grid: bool = False, **init_opts: Any) -> BAResult:
    w, pw = make_w_samples(L, Uhi, N, grid=grid)
    U_pre = build_U_pre(U_fn, A, w)
    return threevar_BA_iterations(X=X, beta1=beta1, beta2=beta2, beta3=beta3,
                                  U_pre=U_pre, pw=pw, tol=tol, maxiter=maxiter, **init_opts)
    
# Add convergence tracking: compute MI and utility per iteration and plot.
# Requirements: matplotlib only (no seaborn), one plot per chart, default colors.

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

def mutual_info_AW_given_O(pagow, pago, pogw, po, pw):
    # I(A;W|O) = sum_x p(x) E_{w|x}[ KL(p(a|x,w) || p(a|x)) ]
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

def objective_value(EU, I_ow, I_aw, I_awgo, beta1, beta2, beta3):
    inv_b1 = 0.0 if (beta1 is None or np.isinf(beta1)) else 1.0 / beta1
    inv_b2 = 0.0 if (beta2 is None or np.isinf(beta2)) else 1.0 / beta2
    inv_b3 = 0.0 if (beta3 is None or np.isinf(beta3)) else 1.0 / beta3
    # General/parallel composite (matches terms in update equations):
    # J = EU - (1/β1) I(X;W) - (1/β2) I(A;W) - (1/β3 - 1/β2) I(A;W|X)
    return EU - inv_b1 * I_ow - inv_b2 * I_aw - (inv_b3 - inv_b2) * I_awgo

def collect_metrics_over_history(history, pw, U_pre, beta1, beta2, beta3):
    rows = []
    for po, pa, pogw, pago, pagow in zip(history["po"], history["pa"], history["pogw"], history["pago"], history["pagow"]):
        pagw = marginalizeo(pogw, pagow)
        I_ow = mutual_info_XW(pogw, po, pw)
        I_ao = mutual_info_AO(pago, pa, po)
        I_awgo = mutual_info_AW_given_O(pagow, pago, pogw, po, pw)
        I_aw = mutual_info_AW(pagw, pa, pw)
        EU = expected_utility(U_pre, pogw, pagow, pw)
        J = objective_value(EU, I_ow, I_aw, I_awgo, beta1, beta2, beta3)
        rows.append(dict(I_ow=I_ow, I_ao=I_ao, I_awgo=I_awgo, I_aw=I_aw, E_U=EU, Objective_value=J))
    df = pd.DataFrame(rows)
    df.index.name = "iteration"
    return df

def plot_convergence(perf_df: pd.DataFrame, xlabel_perf: str = "Iteration"):
    # First chart: mutual informations
    fig1 = plt.figure()
    plt.plot(perf_df.index.values, perf_df["I_ow"], label="I(X;W)")
    plt.plot(perf_df.index.values, perf_df["I_ao"], label="I(A;X)")
    plt.plot(perf_df.index.values, perf_df["I_awgo"], label="I(A;W|X)")
    plt.plot(perf_df.index.values, perf_df["I_aw"], label="I(A;W)")
    plt.xlabel(xlabel_perf)
    plt.ylabel("[nats]")
    plt.legend(loc="lower right")
    plt.title("Convergence of mutual information terms")
    plt.show()
    
    # Second chart: E[U] and Objective
    fig2 = plt.figure()
    plt.plot(perf_df.index.values, perf_df["E_U"], label="E[U]")
    plt.plot(perf_df.index.values, perf_df["Objective_value"], label="Objective J")
    plt.xlabel(xlabel_perf)
    plt.ylabel("[utils]")
    plt.legend(loc="lower right")
    plt.title("Convergence of utility and objective")
    plt.show()
    return fig1, fig2

def show_prob_matrix(M: np.ndarray, name: str = "", indices=None, tol: float = 1e-5):
    M = np.asarray(M)
    cmap = mpl.cm.get_cmap("viridis").copy()
    cmap.set_bad(color="black")  # masked entries will appear black

    def _show_slice(mat2d, title):
        # Mask near-zeros
        masked = np.ma.masked_where(mat2d <= tol, mat2d)
        fig = plt.figure()
        im = plt.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0,
                        aspect="auto", interpolation="nearest")
        plt.colorbar(im)
        plt.title(title)
        plt.xlabel("columns")
        plt.ylabel("rows")
        return fig

    if M.ndim == 2:
        return [_show_slice(M, name or "Matrix")]
    elif M.ndim == 3:
        A, X, N = M.shape
        if indices is None:
            cnt = min(10, N)
            indices = np.unique(np.linspace(0, N-1, cnt, dtype=int))
        figs = []
        for k in indices:
            figs.append(_show_slice(M[:, :, int(k)], f"{name or 'Matrix'} slice k={int(k)}"))
        return figs
    else:
        raise ValueError("Expected a 2D or 3D array.")
    
    
    
# ----------------------------------------------------------------------------
# Slice statistics for p(a|x,w)
# ----------------------------------------------------------------------------
def _even_slice_indices(N: int, k: int = 10) -> np.ndarray:
    """Return up to k evenly spaced integer indices in [0, N-1]."""
    if k >= N:
        return np.arange(N, dtype=int)
    return np.unique(np.linspace(0, N - 1, k, dtype=int))

def compute_pagow_slice_stats(pagow: np.ndarray, indices=None):
    """
    Compute per-slice stats for p(a|x,w).
    Inputs
      pagow:  array of shape (A, X, N)
      indices: optional 1D index list over the last axis (w); if None, picks 10 even slices.
    Returns
      stats: dict with
         - 'indices'        : (K,)
         - 'negentropy01'   : (X, K)   # 1 - H/ln(A)
         - 'maxval'         : (X, K)   # max_a p(a|x,w)
         - 'gap'            : (X, K)   # max_a - second_max_a
         - 'top1'           : (X, K)
         - 'top2'           : (X, K)
         - 'entropy'        : (X, K)   # in nats
      df: tidy pandas DataFrame with columns:
         ['w_index','x','negentropy01','maxval','gap','top1','top2','entropy']
    """
    A, X, N = pagow.shape
    if indices is None:
        indices = _even_slice_indices(N, 10)
    indices = np.asarray(indices, dtype=int)
    # Sanitize: keep only indices within [0, N-1]
    mask_ok = (indices >= 0) & (indices < N)
    if not np.all(mask_ok):
        indices = indices[mask_ok]
    if indices.size == 0:
        raise ValueError(f"No valid slice indices in [0, {N-1}].")
    # allocate
    K = len(indices)
    negentropy01 = np.empty((X, K), dtype=float)
    maxval       = np.empty((X, K), dtype=float)
    gap          = np.empty((X, K), dtype=float)
    top1         = np.empty((X, K), dtype=float)
    top2         = np.empty((X, K), dtype=float)
    entropy      = np.empty((X, K), dtype=float)

    logA = np.log(float(A))
    for kk, j in enumerate(indices):
        P = np.clip(pagow[:, :, j], EPS, 1.0)  # shape (A, X)
        # entropy over actions for each x
        H = -np.sum(P * np.log(P), axis=0)     # (X,)
        entropy[:, kk] = H
        negentropy01[:, kk] = 1.0 - H / (logA + EPS)
        # top-1 and top-2 over actions for each x
        # use partial sort for efficiency
        # argsort descending along axis=0
        order = np.argsort(-P, axis=0)         # (A, X)
        t1 = P[order[0, :], np.arange(X)]
        t2 = P[order[1, :], np.arange(X)]
        top1[:, kk] = t1
        top2[:, kk] = t2
        maxval[:, kk] = t1
        gap[:, kk] = t1 - t2

    # tidy DataFrame
    rows = []
    for kk, j in enumerate(indices):
        for x in range(X):
            rows.append(dict(
                w_index=int(j), x=int(x),
                negentropy01=float(negentropy01[x, kk]),
                maxval=float(maxval[x, kk]),
                gap=float(gap[x, kk]),
                top1=float(top1[x, kk]),
                top2=float(top2[x, kk]),
                entropy=float(entropy[x, kk]),
            ))
    df = pd.DataFrame(rows, columns=["w_index","x","negentropy01","maxval","gap","top1","top2","entropy"])
    return dict(indices=indices, negentropy01=negentropy01, maxval=maxval,
                gap=gap, top1=top1, top2=top2, entropy=entropy), df

def print_pagow_slice_stats_summary(stats: dict):
    """Tiny textual summary over all x and w slices."""
    ne = stats["negentropy01"]; mx = stats["maxval"]; gp = stats["gap"]
    def rng(v): return (float(np.min(v)), float(np.max(v)))
    print("negentropy01 range:", rng(ne))
    print("maxval       range:", rng(mx))
    print("gap          range:", rng(gp))



# # -----------------------------------------------------------------------------
# # FULL-W statistics (curves vs w) for p(a|x,w)
# # -----------------------------------------------------------------------------
# def compute_pagow_stats_over_w(pagow: np.ndarray):
#     """
#     Compute per-x statistics across *all* w (no sub-sampling).
#     Inputs
#       pagow: array (A, X, N) with columns p(a|x,w_j) over a.
#     Returns dict with (X, N) matrices:
#       - 'negentropy01' : 1 - H/ln(A)                 in [0,1]
#       - 'maxval'       : max_a p(a|x,w)             in [0,1]
#       - 'gap'          : top1 - top2                in [0,1]
#       - 'entropy'      : H(p(a|x,w))                in nats
#     """
#     A, X, N = pagow.shape
#     P = np.clip(pagow, 1e-16, 1.0)                 # (A, X, N)
#     H = -np.sum(P * np.log(P), axis=0)             # (X, N)
#     logA = np.log(float(A))
#     negentropy01 = 1.0 - H / (logA + 1e-16)        # (X, N)
#     # top-1 and top-2 along action axis
#     order = np.argsort(-P, axis=0)                 # (A, X, N)
#     top1 = np.take_along_axis(P, order[0:1, :, :], axis=0).squeeze(0)  # (X, N)
#     top2 = np.take_along_axis(P, order[1:2, :, :], axis=0).squeeze(0)  # (X, N)
#     maxval = top1
#     gap = top1 - top2
#     return dict(negentropy01=negentropy01, maxval=maxval, gap=gap, entropy=H)

# def stats_to_df(stats: dict, w: np.ndarray) -> pd.DataFrame:
#     """
#     Convert stats (X,N) matrices to a tidy DataFrame keyed by the actual w values.
#     Columns: ['w','x','negentropy01','maxval','gap','entropy']
#     """
#     w = np.asarray(w).reshape(-1)
#     Nw = w.shape[0]
#     X, N = stats["negentropy01"].shape
#     if N != Nw:
#         raise ValueError(f"Length of w ({Nw}) must match stats width ({N}).")
#     rows = []
#     for x in range(X):
#         for j in range(N):
#             rows.append(dict(
#                 w=float(w[j]), x=int(x),
#                 negentropy01=float(stats["negentropy01"][x, j]),
#                 maxval=float(stats["maxval"][x, j]),
#                 gap=float(stats["gap"][x, j]),
#                 entropy=float(stats["entropy"][x, j]),
#             ))
#     return pd.DataFrame(rows).sort_values("w").reset_index(drop=True)

# def plot_pagow_stats_vs_w_combined(w: np.ndarray, stats: dict, which_x=None, xlabel: str = "w"):
#     """
#     Single figure: plots negentropy01, gap, and maxval as functions of w,
#     with one line per (stat, x). Legend labels like 'negentropy | x=1'.
#     Args:
#       w        : (N,) real vector of w-values (grid or sample order)
#       stats    : dict from compute_pagow_stats_over_w (keys: 'negentropy01','gap','maxval',...)
#       which_x  : iterable of x indices to include (0-based). If None, includes all x.
#     Returns:
#       fig (matplotlib.figure.Figure)
#     """
#     w = np.asarray(w).reshape(-1)
#     neg = np.asarray(stats["negentropy01"])  # (X,N) in [0,1]
#     gap = np.asarray(stats["gap"])           # (X,N) in [0,1]
#     mxv = np.asarray(stats["maxval"])        # (X,N) in [0,1]
#     X, N = neg.shape
#     if N != w.shape[0]:
#         raise ValueError(f"Length of w ({w.shape[0]}) must match stat width ({N}).")
#     if which_x is None:
#         which_x = range(X)
#     fig = plt.figure()
#     for x in which_x:
#         if x == 0:
#         # Three lines for each x on the same axes
#             plt.plot(w, neg[x, :], label=f"negentropy | x={x+1}")
#             #plt.plot(w, gap[x, :], label=f"gap | x={x+1}")
#             #plt.plot(w, mxv[x, :], label=f"max | x={x+1}")
#         else:
#             plt.plot(w, neg[x, :], label=f"negentropy | x={x+1}")
#             #plt.plot(w, gap[x, :], label=f"gap | x={x+1}")
#             #plt.plot(w, mxv[x, :], label=f"max | x={x+1}")
            
#     plt.xlabel(xlabel)
#     plt.ylabel("value")
#     plt.title("negentropy / gap / max vs w (all x)")
#     plt.ylim(0.0, 1.0)  # all three stats are in [0,1]
#     plt.legend(loc="best", ncol=2)
#     return fig

# -----------------------------------------------------------------------------
# FULL-W stats (curves vs w) for posterior p(a|w): negentropy, max, gap
# -----------------------------------------------------------------------------
def compute_pagw_stats_over_w(pagw: np.ndarray):
    """
    Compute statistics of p(a|w) across *all* w (no sub-sampling).
    Input:
      pagw: (A, N) posterior over actions per w
    Returns dict with (N,) arrays:
      - 'negentropy01' : 1 - H(p(a|w))/ln(A)           in [0,1]
      - 'maxval'       : max_a p(a|w)                  in [0,1]
      - 'gap'          : top1 - top2                   in [0,1] (0 if A<2)
      - 'entropy'      : H(p(a|w))                     in nats
    """
    A, N = pagw.shape
    P = np.clip(pagw, EPS, 1.0)                  # (A, N)
    H = -np.sum(P * np.log(P), axis=0)           # (N,)
    logA = np.log(float(A))
    negentropy01 = 1.0 - H / (logA + EPS)        # (N,)
    # top-1 and top-2 along actions for each column (w)
    order = np.argsort(-P, axis=0)               # (A, N)
    top1 = P[order[0, :], np.arange(N)]          # (N,)
    if A >= 2:
        top2 = P[order[1, :], np.arange(N)]      # (N,)
        gap = top1 - top2
    else:
        gap = np.zeros_like(top1)
    maxval = top1
    return dict(negentropy01=negentropy01, maxval=maxval, gap=gap, entropy=H)

def pagw_stats_df(stats: dict, w: np.ndarray) -> pd.DataFrame:
    """
    Tidy DataFrame for p(a|w) stats:
      columns: ['w','negentropy01','maxval','gap','entropy']
    """
    w = np.asarray(w).reshape(-1)
    y_ne = np.asarray(stats["negentropy01"]).reshape(-1)
    y_mx = np.asarray(stats["maxval"]).reshape(-1)
    y_gp = np.asarray(stats["gap"]).reshape(-1)
    y_H  = np.asarray(stats["entropy"]).reshape(-1)
    if w.shape[0] != y_ne.shape[0]:
        raise ValueError(f"Length of w ({w.shape[0]}) must match stats width ({y_ne.shape[0]}).")
    return (
        pd.DataFrame({"w": w, "negentropy01": y_ne, "maxval": y_mx, "gap": y_gp, "entropy": y_H})
          .sort_values("w")
          .reset_index(drop=True)
    )

def plot_pagow_stats_vs_w_combined(w: np.ndarray, stats: dict, xlabel: str = "w"):
    """
    Single figure: plots negentropy01, gap, and maxval of p(a|w) as functions of w.
    Args:
      w     : (N,) real vector of w-values (grid or sample order)
      stats : dict from compute_pagw_stats_over_w (keys: 'negentropy01','gap','maxval')
    Returns:
      fig (matplotlib.figure.Figure)
    """
    w = np.asarray(w).reshape(-1)
    neg = np.asarray(stats["negentropy01"]).reshape(-1)
    gp  = np.asarray(stats["gap"]).reshape(-1)
    mx  = np.asarray(stats["maxval"]).reshape(-1)
    if w.shape[0] != neg.shape[0]:
        raise ValueError(f"Length of w ({w.shape[0]}) must match stats width ({neg.shape[0]}).")
    fig = plt.figure()
    plt.plot(w, neg, label="negentropy | p(a|w)")
    plt.plot(w, gp,  label="gap (top1 - top2)")
    plt.plot(w, mx,  label="max")
    plt.xlabel(xlabel)
    plt.ylabel("value")
    plt.title("negentropy / gap / max of p(a|w) vs w")
    plt.ylim(0.0, 1.0)
    plt.legend(loc="best")
    return fig

def U_fn_quad(a: int, w: float) -> float:
    mu = mu_as[a-1]
    return -0.5 * (w - mu)**2

def U_fn(a: int, w: float) -> float: 
    mu = mu_as[a-1] 
    sigma = sigma_as[a-1] 
    return -0.5*np.log(2*np.pi*sigma**2) - 0.5*((w - mu)**2)/(sigma**2)


# --- Run BA with history tracking on the same quadratic-utility example ---
X = 2
A = 3
mu_as = np.array([-1.0, 0, 1.0])
sigma_as = np.ones(A)*2
epsilon = 0.5
L, Uhi = mu_as[0] - epsilon, mu_as[-1] + epsilon

beta1, beta2, beta3 = 100, 40, 20

# Grid sampling for determinism

n_samples = 300
w, pw = make_w_samples(L, Uhi, n_samples, grid=True)
U_pre = build_U_pre(U_fn, A, w)

res = threevar_BA_iterations(
    X=X, beta1=beta1, beta2=beta2, beta3=beta3,
    U_pre=U_pre, pw=pw, tol=1e-10, maxiter=1000,
    init_pogw_uniformly=False, init_pogw_sparse=True, init_pagow_uniformly=True,
    track_history=True
)

perf_df = collect_metrics_over_history(res.history, pw, U_pre, beta1, beta2, beta3)

# Show the table and plots
print(perf_df.head())

fig1, fig2 = plot_convergence(perf_df)



print("p(x|w):\n", res.pogw)
print("p(x):", res.po)
print("p(a|x,w):\n", res.pagow)
print("p(a|x):\n", res.pago)
print("p(a):", res.pa)
print("p(a|w):", res.pagw)


figs = show_prob_matrix(res.pago, name="p(a|x)")
plt.show()

# Display a 3D matrix with slices


for fig in show_prob_matrix(res.pogw, name="p(x|,w)"):
    plt.show()
    
figs = show_prob_matrix(res.pagw, name="p*(a|w)")
plt.show()


# for i,fig in enumerate(show_prob_matrix(res.pagow, name="p(a|x,w)")):
#     if i % 60 == 0:
#       plt.show()
    
    
# Compute stats on all 10 default slices
stats, stats_df = compute_pagow_slice_stats(res.pagow)

# Or pick explicit slice indices (e.g., first 10 w’s)
idx = list(range(10))
stats, stats_df = compute_pagow_slice_stats(res.pagow, indices=idx)

# Quick summary + preview
print_pagow_slice_stats_summary(stats)
print(stats_df.head())

# --- Full-w curves vs w (combined plot) ---
# requires the actual w grid/points used to build U_pre/pagow
# all_stats = compute_pagow_stats_over_w(res.pagow)
# curves_df = stats_to_df(all_stats, w)   # assumes `w` is available in scope
# fig_all = plot_pagow_stats_vs_w_combined(w, all_stats)  # all x on one plot, all three stats
# plt.show()




# --- Full-w curves vs w (combined plot) ---
# requires the actual w grid/points used to build U_pre/pagow
# all_stats = compute_pagow_stats_over_w(res.pagow)
# curves_df = stats_to_df(all_stats, w)   # assumes `w` is available in scope
# fig_all = plot_pagow_stats_vs_w_combined(w, all_stats)  # all x on one plot, all three stats
# plt.show()

# --- Negentropy curve for posterior p(a|w) ---
# --- p(a|w) stats vs w (combined plot) ---
pagw_stats = compute_pagw_stats_over_w(res.pagw)
pagw_df    = pagw_stats_df(pagw_stats, w)     # optional: inspect or save
fig_all    = plot_pagow_stats_vs_w_combined(w, pagw_stats)
plt.show()

# Optional: tidy table
df = pagw_stats_df(pagw_stats, w)
print(df.head())

# # --- Full-w curves vs w (combined plot) ---
# # requires the actual w grid/points used to build U_pre/pagow
# all_stats = compute_pagow_stats_over_w(res.pagow)
# curves_df = stats_to_df(all_stats, w)   # assumes `w` is available in scope
# fig_all = plot_pagow_stats_vs_w_combined(w, all_stats)  # all x on one plot, all three stats
# plt.show()