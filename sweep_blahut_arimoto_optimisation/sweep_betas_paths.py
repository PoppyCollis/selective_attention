import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List, Set
from tqdm.auto import tqdm
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection

from blahut_arimoto import (
    make_w_samples, build_U_pre, U_fn, marginalizeo,
    mutual_info_XW, mutual_info_AO, mutual_info_WA_given_O,
    expected_utility, threevar_BA_iterations_warm
)

# ----------------------- Config & problem -----------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "results"
ART_DIR = OUT_DIR / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)
out_csv = OUT_DIR / "sweep_betas_results_paths.csv"


def plot_J_landscape(csv_path: str,
                     kind: str = "surface",
                     cmap: str = "viridis",
                     elev: float = 30,
                     azim: float = 45,
                     figsize=(8,6)) -> None:
    """
    Plot the J landscape with β2 on x–axis, β3 on y–axis and J on z–axis.

    Parameters
    ----------
    csv_path : str
        Path to the results CSV (must contain columns: beta2, beta3, J_best).
    kind : {'surface','heatmap'}
        'surface' for 3-D surface, 'heatmap' for 2-D color map.
    cmap : str
        Matplotlib colormap.
    elev, azim : float
        Elevation and azimuth of the 3-D view (only for 'surface').
    figsize : tuple
        Figure size.
    """
    df = pd.read_csv(csv_path)
    if not {'beta2','beta3','J_best'}.issubset(df.columns):
        raise ValueError("CSV must contain columns: beta2, beta3, J_best")

    # create pivoted grid (assumes regular grid or repeated coords)
    pivot = df.pivot_table(index='beta3', columns='beta2', values='J_best', aggfunc='mean')
    X = pivot.columns.values
    Y = pivot.index.values
    Xg, Yg = np.meshgrid(X, Y)
    Z = pivot.values

    if kind == "surface":
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(Xg, Yg, Z, cmap=cmap, edgecolor='none')
        ax.set_xlabel(r"$\beta_2$")
        ax.set_ylabel(r"$\beta_3$")
        ax.set_zlabel(r"$J$")
        ax.view_init(elev=elev, azim=azim)
        fig.colorbar(surf, shrink=0.6, aspect=12, label=r"$J$")
        plt.tight_layout()
        plt.show()

    elif kind == "heatmap":
        fig, ax = plt.subplots(figsize=figsize)
        c = ax.pcolormesh(Xg, Yg, Z, cmap=cmap, shading='auto')
        ax.set_xlabel(r"$\beta_2$")
        ax.set_ylabel(r"$\beta_3$")
        fig.colorbar(c, ax=ax, label=r"$J$")
        plt.tight_layout()
        plt.show()

    else:
        raise ValueError("kind must be 'surface' or 'heatmap'")

# Log-space bounds for (beta2, beta3)
BMIN, BMAX = 0.1, 10.0
LBMIN, LBMAX = np.log10(BMIN), np.log10(BMAX)

INVALID_TOL = 1e-5  # tolerance for identifying invalid solutions

# Problem mirrors optimise_for_config.py
X = 3
A = 3
mu_as = np.array([-96, 0, 96])
sigma_as = np.ones(A) * 64
epsilon = 64
L, Uhi = mu_as[0] - epsilon, mu_as[-1] + epsilon

beta1 = np.inf
n_samples = 500
w, pw = make_w_samples(L, Uhi, n_samples, grid=True)
U_pre = build_U_pre(U_fn, A, w, mu_as, sigma_as)

# BA controls
maxiter = 200
tol = 1e-9
perturb_std = 1e-3  # noise on warm starts

# Multi-start
k0 = 3      # fresh diversified restarts
k_warm = 2  # warm-started restarts
rng = np.random.default_rng(123)

# ----------------------- Helpers -----------------------

def clamp_beta(b: float) -> float:
    return float(np.clip(b, BMIN, BMAX))

def components_from_result(beta2, beta3, res):
    pagw = marginalizeo(res.pogw, res.pagow)
    I_ow = float(mutual_info_XW(res.pogw, res.po, pw))
    I_ao = float(mutual_info_AO(res.pago, res.pa, res.po))
    I_wago = float(mutual_info_WA_given_O(res.pagow, res.pago, res.pogw, res.po, pw))
    EU = float(expected_utility(U_pre, res.pogw, res.pagow, pw))
    J = EU - (0.0 if np.isinf(beta1) else (1.0/beta1)*I_ow) \
          - (1.0/beta2)*I_ao - (1.0/beta3)*I_wago
    return dict(J=float(J), EU=EU, I_ow=I_ow, I_ao=I_ao,
                I_wago=I_wago, pagw=pagw)

def run_one(beta2: float, beta3: float,
            warm_inits: Optional[List[Tuple[np.ndarray,np.ndarray]]] = None,
            n_fresh: int = 0, n_warm: int = 0):
    runs = []

    # Fresh starts
    for _ in range(n_fresh):
        res = threevar_BA_iterations_warm(
            X=X, beta1=beta1, beta2=beta2, beta3=beta3,
            U_pre=U_pre, pw=pw, tol=tol, maxiter=maxiter,
            init_pogw=None, init_pagow=None,
            init_pogw_uniformly=False, init_pogw_sparse=True,
            init_pagow_uniformly=True, perturb_std=0.0, track_history=False
        )
        comp = components_from_result(beta2, beta3, res)
        runs.append((comp, res))

    # Warm starts
    if warm_inits:
        for (pogw0, pagow0) in warm_inits[:n_warm]:
            res = threevar_BA_iterations_warm(
                X=X, beta1=beta1, beta2=beta2, beta3=beta3,
                U_pre=U_pre, pw=pw, tol=tol, maxiter=maxiter,
                init_pogw=pogw0, init_pagow=pagow0,
                init_pogw_uniformly=False, init_pogw_sparse=False,
                init_pagow_uniformly=False, perturb_std=perturb_std,
                track_history=False
            )
            comp = components_from_result(beta2, beta3, res)
            runs.append((comp, res))

    runs.sort(key=lambda t: -t[0]["J"])
    comps = [c for (c, _) in runs]
    best_comp, best_res = runs[0]

    # Aggregate scalars
    stats = dict(
        J_best=best_comp["J"],
        J_mean=float(np.mean([c["J"] for c in comps])),
        J_std=float(np.std([c["J"] for c in comps])),
        EU_best=best_comp["EU"],
        EU_mean=float(np.mean([c["EU"] for c in comps])),
        I_ow_best=best_comp["I_ow"],
        I_ow_mean=float(np.mean([c["I_ow"] for c in comps])),
        I_ao_best=best_comp["I_ao"],
        I_ao_mean=float(np.mean([c["I_ao"] for c in comps])),
        I_wago_best=best_comp["I_wago"],
        I_wago_mean=float(np.mean([c["I_wago"] for c in comps])),
        n_runs=len(runs),
    )
    return stats, best_res, runs

def save_artifacts(beta2: float, beta3: float, best_res, stats: Dict) -> Dict:
    pagw = components_from_result(beta2, beta3, best_res)["pagw"]
    pagw_path = ART_DIR / f"pagw_b2={beta2:.6f}_b3={beta3:.6f}.npy"
    np.save(pagw_path, pagw)
    stats = dict(stats)
    stats.update(dict(beta2=float(beta2), beta3=float(beta3),
                      pagw_path=str(pagw_path)))
    return stats

# Memoization for evaluated points (avoid recompute)
evaluated: Dict[Tuple[float,float], Dict] = {}
best_state_at: Dict[Tuple[float,float], Dict[str,np.ndarray]] = {}

def eval_point(beta2: float, beta3: float) -> Dict:
    key = (float(beta2), float(beta3))
    if key in evaluated:
        return evaluated[key]
    # warm-start pool: nearest already-seen points (up to K)
    K = 6
    if best_state_at:
        def dist2(p):
            (b2, b3) = p
            return (np.log10(beta2)-np.log10(b2))**2 + \
                   (np.log10(beta3)-np.log10(b3))**2
        neigh_keys = sorted(best_state_at.keys(), key=dist2)[:K]
        warm_pool = [(best_state_at[k]["pogw"],
                      best_state_at[k]["pagow"]) for k in neigh_keys]
    else:
        warm_pool = []

    stats, best_res, _ = run_one(beta2, beta3,
                                 warm_inits=warm_pool,
                                 n_fresh=k0, n_warm=k_warm)
    stats = save_artifacts(beta2, beta3, best_res, stats)
    evaluated[key] = stats
    best_state_at[key] = dict(pogw=best_res.pogw.copy(),
                              pagow=best_res.pagow.copy())
    return stats

# ----------------------- Paths -----------------------

def linspace_log(a: float, b: float, n: int) -> np.ndarray:
    la, lb = np.log10(a), np.log10(b)
    return 10**np.linspace(la, lb, n)

def axis_paths(n_seeds: int = 3, n_steps: int = 15) -> List[List[Tuple[float,float]]]:
    mid = 10**((LBMIN + LBMAX)/2.0)
    bvals = linspace_log(BMIN, BMAX, n_seeds)
    paths = []
    for b2 in bvals:  # vertical
        paths.append([(b2, b3) for b3 in linspace_log(BMIN, BMAX, n_steps)])
    for b3 in bvals:  # horizontal
        paths.append([(b2, b3) for b2 in linspace_log(BMIN, BMAX, n_steps)])
    return paths

def diagonal_paths(n_steps: int = 15) -> List[List[Tuple[float,float]]]:
    diag = [(b, b) for b in linspace_log(BMIN, BMAX, n_steps)]
    mid_log = (LBMIN + LBMAX)/2.0
    offs = np.linspace(-(LBMAX-LBMIN)/2.0, (LBMAX-LBMIN)/2.0, n_steps)
    anti = []
    for d in offs:
        lb2 = mid_log + d
        lb3 = mid_log - d
        anti.append((10**lb2, 10**lb3))
    return [diag, anti]

def greedy_hill_path(start: Tuple[float,float], step_log: float = 0.15,
                     max_steps: int = 40, patience: int = 5) -> List[Tuple[float,float]]:
    """Follow local improvements in J using a small log-space stencil."""
    cur = start
    best = eval_point(*cur)
    no_improve = 0
    path = [cur]
    while len(path) < max_steps and no_improve < patience:
        lb2, lb3 = np.log10(cur[0]), np.log10(cur[1])
        candidates = []
        deltas = [(step_log,0),(0,step_log),(-step_log,0),(0,-step_log),
                  (step_log,step_log),(-step_log,-step_log),
                  (step_log,-step_log),(-step_log,step_log)]
        for (db2, db3) in deltas:
            nb2 = clamp_beta(10**(lb2+db2))
            nb3 = clamp_beta(10**(lb3+db3))
            candidates.append((nb2, nb3))
        vals = [(eval_point(b2,b3)["J_best"], (b2,b3))
                for (b2,b3) in candidates]
        vals.sort(reverse=True)
        if vals[0][0] > best["J_best"] + 1e-12:
            best = eval_point(*vals[0][1])
            cur = vals[0][1]
            path.append(cur)
            no_improve = 0
        else:
            no_improve += 1
    return path

# ----------------------- Driver -----------------------

def run_hybrid(paths_axis=True, paths_diag=True,
               n_axis_seeds=3, n_axis_steps=15,
               do_greedy=True, greedy_starts: int = 4):
    planned = 0
    all_paths = []
    if paths_axis:
        P = axis_paths(n_seeds=n_axis_seeds, n_steps=n_axis_steps)
        all_paths += P
        planned += sum(len(p) for p in P)
    if paths_diag:
        P = diagonal_paths(n_steps=n_axis_steps)
        all_paths += P
        planned += sum(len(p) for p in P)

    pbar = tqdm(total=planned, desc="Path evaluations")
    visited: Set[Tuple[float,float]] = set()
    for path in all_paths:
        for (b2,b3) in path:
            key = (float(b2), float(b3))
            if key in visited:
                pbar.update(1)
                continue
            eval_point(b2, b3)
            visited.add(key)
            pbar.update(1)
    pbar.close()

    if do_greedy:
        starts = []
        for _ in range(greedy_starts):
            lb2 = rng.uniform(LBMIN, LBMAX)
            lb3 = rng.uniform(LBMIN, LBMAX)
            starts.append((10**lb2, 10**lb3))
        for s in tqdm(starts, desc="Greedy starts"):
            greedy_hill_path(s)

    df = pd.DataFrame(list(evaluated.values()))
    df = df.sort_values(["beta2","beta3"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv.resolve())
    print(df.head())
    
    plot_J_landscape("results/sweep_betas_results_paths.csv", kind="surface")


if __name__ == "__main__":
    print("Writing to:", out_csv)
    run_hybrid(paths_axis=True, paths_diag=True,
               n_axis_seeds=3, n_axis_steps=15,
               do_greedy=True, greedy_starts=4)
    

