import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

EPS = 1e-16

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
    cmap = mpl.colormaps.get_cmap("viridis").copy()
    #cmap.set_bad(color="black")  # masked entries will appear black

    def _show_slice(mat2d, title):
        # Mask near-zeros
        #masked = np.ma.masked_where(mat2d <= tol, mat2d)
        fig = plt.figure()
        im = plt.imshow(mat2d, cmap=cmap, vmin=0.0, vmax=1.0,
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
    order = np.argsort(-P, axis=0)               # (A, N)
    top1 = P[order[0, :], np.arange(N)]          # (N,)
    if A >= 2:
        top2 = P[order[1, :], np.arange(N)]      # (N,)
        gap = top1 - top2
    else:
        gap = np.zeros_like(top1)
    maxval = top1
    # Min–max normalize maxval across w: min -> 0, max -> 1
    _mn_mx = float(np.min(maxval))
    _mx_mx = float(np.max(maxval))
    if _mx_mx > _mn_mx + 1e-12:
        maxval_minmax = (maxval - _mn_mx) / (_mx_mx - _mn_mx)
    else:
        maxval_minmax = np.zeros_like(maxval)
    return dict(negentropy01=negentropy01, maxval=maxval_minmax, gap=gap, entropy=H)

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
    #plt.plot(w, gp,  label="gap (top1 - top2)")
    #plt.plot(w, mx,  label="max")
    plt.xlabel(xlabel)
    plt.ylabel("value")
    plt.title("confidence of p(a|w) vs w")
    plt.ylim(0.0, 1.0)
    plt.legend(loc="best")
    return fig