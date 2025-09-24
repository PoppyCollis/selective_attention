
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; needed for 3-D projection
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

# Make sure these are set before launching workers (prevents oversubscription)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from blahut_arimoto import (make_w_samples,
                            build_U_pre, 
                            U_fn, 
                            threevar_BA_iterations_warm, 
                            mutual_info_XW, 
                            mutual_info_AO, 
                            mutual_info_WA_given_O,
                            expected_utility, 
                            marginalizeo)

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "results"
ART_DIR = OUT_DIR / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)
out_csv = OUT_DIR / "sweep_betas_results.csv"

print("CWD:", os.getcwd())
print("Writing CSV to:", out_csv)

POST_DIR = Path("results") / "posteriors"
POST_DIR.mkdir(parents=True, exist_ok=True)

# --- add these helpers next to your imports ---

def components_from_result(U_pre, pw, beta1, beta2, beta3, res):
    """Return EU and MI components + J for a BAResult."""
    pagw = marginalizeo(res.pogw, res.pagow)  # shape (A, W) aggregated over X
    I_ow = mutual_info_XW(res.pogw, res.po, pw)
    I_ao = mutual_info_AO(res.pago, res.pa, res.po)
    I_wago = mutual_info_WA_given_O(res.pagow, res.pago, res.pogw, res.po, pw)
    EU = expected_utility(U_pre, res.pogw, res.pagow, pw)
    # J matches your definition (beta1==inf kills the I_ow term inside J)
    J = float(EU - (0.0 if np.isinf(beta1) else (1.0/beta1)*I_ow)
                   - (1.0/beta2)*I_ao - (1.0/beta3)*I_wago)
    return dict(J=J, EU=float(EU), I_ow=float(I_ow), I_ao=float(I_ao), I_wago=float(I_wago), pagw=pagw)

def plot_J_landscape(csv_path: str,
                     kind: str = "surface",
                     cmap: str = "viridis",
                     elev: float = 30,
                     azim: float = 45,
                     figsize=(8,6),
                     save_paths: bool = False,
                     fig_path: str = "J_landscape_fig.pkl",
                     png_path: str = "J_landscape.png") -> None:
    """
    Plot the J landscape with β2 on x–axis, β3 on y–axis and J on z–axis,
    and optionally save both a static PNG and a pickled Matplotlib figure.

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
    save_paths : bool
        If True, save both a PNG (png_path) and a pickled figure (fig_path).
    fig_path : str
        Where to store the pickled Matplotlib figure.
    png_path : str
        Where to store the static PNG snapshot.
    """
    df = pd.read_csv(csv_path)
    if not {'beta2','beta3','J_best'}.issubset(df.columns):
        raise ValueError("CSV must contain columns: beta2, beta3, J_best")

    pivot = df.pivot_table(index='beta3', columns='beta2',
                           values='J_best', aggfunc='mean')
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
    elif kind == "heatmap":
        fig, ax = plt.subplots(figsize=figsize)
        c = ax.pcolormesh(Xg, Yg, Z, cmap=cmap, shading='auto')
        ax.set_xlabel(r"$\beta_2$")
        ax.set_ylabel(r"$\beta_3$")
        fig.colorbar(c, ax=ax, label=r"$J$")
    else:
        raise ValueError("kind must be 'surface' or 'heatmap'")

    plt.tight_layout()

    if save_paths:
        # static image
        fig.savefig(png_path, dpi=300)
        # pickle the live figure object for later re-rendering
        with open(fig_path, "wb") as f:
            pickle.dump(fig, f)
        print(f"Saved PNG to {png_path}")
        print(f"Saved pickled figure to {fig_path}")

    plt.show()


config = 1
exp = 1
card_x = 3

outfile_name = "sweep_betas_results" + f"{config}" + f"{exp}" + f"{card_x}"
# Problem setup (mirrors your optimise_for_config.py) ---
X = card_x
A = 3
mu_as = np.array([-96, 0, 96])
sigma_as = np.ones(A) * 64
epsilon = 64
L, Uhi = mu_as[0] - epsilon, mu_as[-1] + epsilon

beta1 = np.inf
n_samples = 300
w, pw = make_w_samples(L, Uhi, n_samples, grid=True)
U_pre = build_U_pre(U_fn, A, w, mu_as, sigma_as)

# --- Sweep config ---
beta_min, beta_max = 0.4, 4
G = 30  # grid resolution per axis
grid_betas = [(b2, b3) for b2 in np.linspace(beta_min, beta_max, G)
                        for b3 in np.linspace(beta_min, beta_max, G)]

# Multi-start settings per grid point
k0 = 3      # diversified fresh starts
k_warm = 2  # warm-started runs per point
k_max = k0 + k_warm
maxiter = 100   
tol = 1e-9
perturb_std = 1e-3  # tiny noise on warm-started states

# Output (relative path; will be created if missing)
out_csv = os.path.join("results", outfile_name)
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
flush_every = 20  # write partial CSV every N grid points

def neighbors(idx, k=4):
    # simple recency-based neighbor pool; swap with true kNN if desired
    M = 10
    start = max(0, idx - M)
    return list(range(start, idx))

records = []
best_state_at = {}  # (b2,b3) -> dict(pogw=..., pagow=...)


# worker must be top-level (picklable)
def _ba_worker(task):
    """
    One restart for a given (beta2, beta3).
    task: dict with keys:
        beta2, beta3, X, beta1, U_pre, pw, tol, maxiter,
        init_pogw, init_pagow, init_pogw_uniformly, init_pogw_sparse,
        init_pagow_uniformly, perturb_std
    Returns (comp_dict, light_state) where light_state has pogw/pagow for warm starts.
    """
    beta2 = task["beta2"]; beta3 = task["beta3"]
    res = threevar_BA_iterations_warm(
        X=task["X"], beta1=task["beta1"], beta2=beta2, beta3=beta3,
        U_pre=task["U_pre"], pw=task["pw"], tol=task["tol"], maxiter=task["maxiter"],
        init_pogw=task["init_pogw"], init_pagow=task["init_pagow"],
        init_pogw_uniformly=task["init_pogw_uniformly"],
        init_pogw_sparse=task["init_pogw_sparse"],
        init_pagow_uniformly=task["init_pagow_uniformly"],
        perturb_std=task["perturb_std"], track_history=False
    )
    pagw = marginalizeo(res.pogw, res.pagow)
    I_ow = float(mutual_info_XW(res.pogw, res.po, task["pw"]))
    I_ao = float(mutual_info_AO(res.pago, res.pa, res.po))
    I_wago = float(mutual_info_WA_given_O(res.pagow, res.pago, res.pogw, res.po, task["pw"]))
    EU = float(expected_utility(task["U_pre"], res.pogw, res.pagow, task["pw"]))
    J = EU - (0.0 if math.isinf(task["beta1"]) else (1.0/task["beta1"])*I_ow) \
          - (1.0/task["beta2"])*I_ao - (1.0/task["beta3"])*I_wago
    comp = dict(J=float(J), EU=EU, I_ow=I_ow, I_ao=I_ao, I_wago=I_wago, pagw=pagw)
    light = dict(pogw=res.pogw, pagow=res.pagow)  # used only if this run becomes “best”
    return comp, light

def run_one_parallel(beta2: float, beta3: float,
                     warm_inits: Optional[list] = None,
                     n_fresh: int = 0, n_warm: int = 0,
                     max_workers: Optional[int] = None):
    """Parallel restarts for a single (beta2, beta3)."""
    tasks = []

    # Fresh restarts
    for _ in range(n_fresh):
        tasks.append(dict(
            beta2=beta2, beta3=beta3,
            X=X, beta1=beta1, U_pre=U_pre, pw=pw,
            tol=tol, maxiter=maxiter,
            init_pogw=None, init_pagow=None,
            init_pogw_uniformly=False, init_pogw_sparse=True,
            init_pagow_uniformly=True, perturb_std=0.0
        ))

    # Warm starts (slight perturbation to escape inherited basins)
    warm_inits = warm_inits or []
    for (pogw0, pagow0) in warm_inits[:n_warm]:
        tasks.append(dict(
            beta2=beta2, beta3=beta3,
            X=X, beta1=beta1, U_pre=U_pre, pw=pw,
            tol=tol, maxiter=maxiter,
            init_pogw=pogw0, init_pagow=pagow0,
            init_pogw_uniformly=False, init_pogw_sparse=False,
            init_pagow_uniformly=False, perturb_std=perturb_std
        ))

    runs = []
    if len(tasks) == 0:
        return dict(J_best=-np.inf, J_mean=np.nan, J_std=np.nan, EU_best=np.nan,
                    EU_mean=np.nan, I_ow_best=np.nan, I_ow_mean=np.nan,
                    I_ao_best=np.nan, I_ao_mean=np.nan,
                    I_wago_best=np.nan, I_wago_mean=np.nan,
                    n_runs=0), None, []

    # fork processes; set workers ~= physical cores
    with ProcessPoolExecutor(max_workers=max_workers or os.cpu_count()) as ex:
        futures = [ex.submit(_ba_worker, t) for t in tasks]
        for fut in as_completed(futures):
            comp, light = fut.result()
            runs.append((comp, light))

    # pick best by J
    runs.sort(key=lambda t: -t[0]["J"])
    comps = [c for (c, _) in runs]
    best_comp, best_light = runs[0]

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
    return stats, best_light, runs


try:
    pbar = tqdm(total=len(grid_betas), desc="Sweeping (β2, β3)")
    for idx, (b2, b3) in enumerate(grid_betas):
        # Warm-start candidates from recent neighbors
        warm_pool = []
        for j in neighbors(idx):
            key = grid_betas[j]
            if key in best_state_at:
                res_state = best_state_at[key]
                warm_pool.append((res_state["pogw"], res_state["pagow"]))

        stats, best_light, runs = run_one_parallel(
            b2, b3, warm_inits=warm_pool, n_fresh=k0, n_warm=k_warm,
            max_workers=None  # or an explicit int
        )
        records.append(dict(beta2=float(b2), beta3=float(b3), **stats))
    
        # ---- save posterior arrays for this beta2,beta3 ----
        fname_base = f"b2={b2:.6f}_b3={b3:.6f}"

        # p(s|x)-style marginal
        pagw = marginalizeo(best_light["pogw"], best_light["pagow"])

        np.save(POST_DIR / f"pagw_{fname_base}.npy", pagw)
        np.save(POST_DIR / f"pogw_{fname_base}.npy", best_light["pogw"])
        np.save(POST_DIR / f"pagow_{fname_base}.npy", best_light["pagow"])

        # ----------------------------------------------------


        # cache best state for warm starts
        best_state_at[(b2, b3)] = dict(
            pogw=best_light["pogw"].copy(),
            pagow=best_light["pagow"].copy()
        )
        # periodic flush
        if (idx + 1) % flush_every == 0:
            df_partial = pd.DataFrame(records)
            df_partial.to_csv(out_csv, index=False)

        pbar.update(1)
    pbar.close()

    df = pd.DataFrame(records).sort_values(['beta2', 'beta3']).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")
    
except KeyboardInterrupt:
    # save what we have on interrupt
    df_partial = pd.DataFrame(records)
    df_partial.to_csv(out_csv, index=False)
    print(f"\nInterrupted. Partial results saved to: {out_csv}")
    raise


plot_J_landscape("results/sweep_betas_results.csv",
                     kind= "surface",
                     save_paths= True,
                     fig_path = "J_landscape_fig_config3.pkl",
                     png_path = "J_landscape_fig_config3.png")
    
# plot_J_landscape("results/sweep_betas_results.csv", kind="surface")
