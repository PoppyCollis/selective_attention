# sweep_betas_discard_invalid.py
import os
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D)
import pickle

# Prevent BLAS oversubscription when using process-level parallelism
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ---- Import your BA + utilities ----
from blahut_arimoto import (
    make_w_samples,
    build_U_pre,
    U_fn,
    threevar_BA_iterations_warm,  # if your warm-capable BA is in another module, import from there
    mutual_info_XW,
    mutual_info_AO,
    mutual_info_WA_given_O,
    expected_utility,
    marginalizeo,
)

# ====================== Plot helper ======================
def plot_J_landscape(csv_path: Path,
                     kind: str = "surface",
                     cmap: str = "viridis",
                     elev: float = 30,
                     azim: float = 45,
                     figsize=(8, 6),
                     save_paths: bool = False,
                     fig_path: Path = Path("J_landscape_fig.pkl"),
                     png_path: Path = Path("J_landscape.png")) -> None:
    df = pd.read_csv(csv_path)
    if not {'beta2','beta3','J_best'}.issubset(df.columns):
        raise ValueError("CSV must contain columns: beta2, beta3, J_best")

    pivot = df.pivot_table(index='beta3', columns='beta2', values='J_best', aggfunc='mean')
    Xg, Yg = np.meshgrid(pivot.columns.values, pivot.index.values)
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
        fig.savefig(png_path, dpi=300)
        with open(fig_path, "wb") as f:
            pickle.dump(fig, f)
        print(f"Saved PNG to {png_path}")
        print(f"Saved pickled figure to {fig_path}")
    plt.show()

# ================== Parallel worker ==================
def _ba_worker(task):
    """Run one BA restart and return (comp, light)."""
    res = threevar_BA_iterations_warm(
        X=task["X"], beta1=task["beta1"], beta2=task["beta2"], beta3=task["beta3"],
        U_pre=task["U_pre"], pw=task["pw"], tol=task["tol"], maxiter=task["maxiter"],
        init_pogw=task["init_pogw"], init_pagow=task["init_pagow"],
        init_pogw_uniformly=task["init_pogw_uniformly"],
        init_pogw_sparse=task["init_pogw_sparse"],
        init_pagow_uniformly=task["init_pagow_uniformly"],
        perturb_std=task["perturb_std"], track_history=False
    )
    # Scalars
    I_ow = float(mutual_info_XW(res.pogw, res.po, task["pw"]))
    I_ao = float(mutual_info_AO(res.pago, res.pa, res.po))
    I_wago = float(mutual_info_WA_given_O(res.pagow, res.pago, res.pogw, res.po, task["pw"]))
    EU = float(expected_utility(task["U_pre"], res.pogw, res.pagow, task["pw"]))
    J = EU - (0.0 if math.isinf(task["beta1"]) else (1.0/task["beta1"])*I_ow) \
          - (1.0/task["beta2"])*I_ao - (1.0/task["beta3"])*I_wago
    comp = dict(J=float(J), EU=EU, I_ow=I_ow, I_ao=I_ao, I_wago=I_wago)
    light = dict(pogw=res.pogw, pagow=res.pagow)  # for warm-starts & saving posteriors
    return comp, light

def _make_task(beta2, beta3, *, kind="fresh", pogw0=None, pagow0=None, pert_std=0.0):
    """Prepare a restart task dict."""
    t = dict(
        beta2=beta2, beta3=beta3,
        X=X, beta1=beta1, U_pre=U_pre, pw=pw,
        tol=tol, maxiter=maxiter,
        init_pogw=None, init_pagow=None,
        init_pogw_uniformly=False, init_pogw_sparse=True,
        init_pagow_uniformly=True, perturb_std=pert_std,
    )
    if kind == "warm" and pogw0 is not None and pagow0 is not None:
        t.update(dict(
            init_pogw=pogw0, init_pagow=pagow0,
            init_pogw_uniformly=False, init_pogw_sparse=False,
            init_pagow_uniformly=False, perturb_std=pert_std
        ))
    elif kind == "fresh_rand_pagow":
        # Diversify pago when retrying
        t.update(dict(init_pagow_uniformly=False, init_pogw_sparse=False))
    return t

def run_one_parallel(beta2: float, beta3: float,
                     warm_inits: Optional[list] = None,
                     n_fresh: int = 0, n_warm: int = 0,
                     max_workers: Optional[int] = None,
                     max_extra: int = 12):
    """
    Parallel restarts for a single (beta2, beta3).
    Automatically discards any run with |J - EU| <= INVALID_TOL, and retries up to max_extra times.
    Returns: (stats, best_light) where stats has only scalar summaries plus counters.
    """
    warm_inits = warm_inits or []
    runs = []

    def _submit_collect(tasks):
        nonlocal runs
        if not tasks: return
        with ProcessPoolExecutor(max_workers=max_workers or os.cpu_count()) as ex:
            futs = [ex.submit(_ba_worker, t) for t in tasks]
            for fut in as_completed(futs):
                comp, light = fut.result()
                runs.append((comp, light))

    # First wave
    tasks = []
    for _ in range(n_fresh):
        tasks.append(_make_task(beta2, beta3, kind="fresh", pert_std=0.0))
    for (pogw0, pagow0) in warm_inits[:n_warm]:
        tasks.append(_make_task(beta2, beta3, kind="warm", pogw0=pogw0, pagow0=pagow0, pert_std=perturb_std))
    _submit_collect(tasks)

    # Retry if all invalid (EU≈J)
    extra = 0
    escalate = [(2*perturb_std, "fresh"),
                (2*perturb_std, "warm"),
                (4*perturb_std, "fresh_rand_pagow"),
                (0.0, "fresh_rand_pagow")]
    esc_idx = 0

    def _valid_pairs(pairs):
        return [(c, l) for (c, l) in pairs if abs(c["J"] - c["EU"]) > INVALID_TOL]

    valids = _valid_pairs(runs)
    while not valids and extra < max_extra:
        new_tasks = []
        for _ in range(3):  # add a small batch each escalation step
            pert, kind = escalate[esc_idx % len(escalate)]
            esc_idx += 1
            if kind == "warm" and warm_inits:
                pogw0, pagow0 = warm_inits[np.random.randint(0, len(warm_inits))]
                new_tasks.append(_make_task(beta2, beta3, kind="warm", pogw0=pogw0, pagow0=pagow0, pert_std=pert))
            else:
                new_tasks.append(_make_task(beta2, beta3, kind=kind, pert_std=pert))
            extra += 1
            if extra >= max_extra:
                break
        _submit_collect(new_tasks)
        valids = _valid_pairs(runs)

    # Choose pool = valid runs if any, else all runs
    pool = valids if valids else runs
    if not pool:
        # Shouldn't happen unless something is broken
        stats = dict(J_best=np.nan, J_mean=np.nan, J_std=np.nan,
                     EU_best=np.nan, EU_mean=np.nan,
                     I_ow_best=np.nan, I_ow_mean=np.nan,
                     I_ao_best=np.nan, I_ao_mean=np.nan,
                     I_wago_best=np.nan, I_wago_mean=np.nan,
                     n_runs=0, n_valid=0, n_invalid=0, used_valid=False)
        return stats, None

    pool.sort(key=lambda t: -t[0]["J"]) # sort all of the valid entries by their value of J
    best_comp, best_light = pool[0]

    # Aggregate on pool so that we can get mean and std metrics
    J_list   = [c["J"] for (c, _) in pool]
    EU_list  = [c["EU"] for (c, _) in pool]
    Iow_list = [c["I_ow"] for (c, _) in pool]
    Iao_list = [c["I_ao"] for (c, _) in pool]
    Iwg_list = [c["I_wago"] for (c, _) in pool]

    # Count invalids in the original runs set
    n_invalid = sum(1 for (c, _) in runs if abs(c["J"] - c["EU"]) <= INVALID_TOL)
    n_valid = len(runs) - n_invalid

    stats = dict(
        J_best=float(best_comp["J"]),
        J_mean=float(np.mean(J_list)),
        J_std=float(np.std(J_list)),
        EU_best=float(best_comp["EU"]),
        EU_mean=float(np.mean(EU_list)),
        I_ow_best=float(best_comp["I_ow"]),
        I_ow_mean=float(np.mean(Iow_list)),
        I_ao_best=float(best_comp["I_ao"]),
        I_ao_mean=float(np.mean(Iao_list)),
        I_wago_best=float(best_comp["I_wago"]),
        I_wago_mean=float(np.mean(Iwg_list)),
        n_runs=len(runs),
        n_valid=int(n_valid),
        n_invalid=int(n_invalid),
        used_valid=bool(valids),
    )
    return stats, best_light

def neighbors(idx, k=4):
    # given the position of the beta_2, beta_3 tuple in the grid_betas list
    # take the previous 2 neibouring pairs
    M = 10
    start = max(0, idx - M)
    return list(range(start, idx))

# ========================= Config =========================
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "results"
ART_DIR = OUT_DIR / "artifacts"
POST_DIR = OUT_DIR / "posteriors"
for d in (OUT_DIR, ART_DIR, POST_DIR):
    d.mkdir(parents=True, exist_ok=True)

# File names
outfile_stem = "sweep_betas_results"
out_csv = OUT_DIR / f"{outfile_stem}.csv"

# Reject runs where EU ~ J (invalid/unstable signature)
INVALID_TOL = 1e-5 # we know when EU = J, this is the invalid unstable solution

# Problem (mirror your config)
X = 2  # cardinality of X
A = 3 # cardinality of actions (number of categories to choose from)
mu_as = np.array([-96, -59, 96]) 
sigma_as = np.ones(A) * 64
epsilon = 64
L, Uhi = mu_as[0] - epsilon, mu_as[-1] + epsilon

beta1 = np.inf # always fix at infinity
n_samples = 300 # number of ws in categorical posterior distribution p(a|w)
w, pw = make_w_samples(L, Uhi, n_samples, grid=True) 
U_pre = build_U_pre(U_fn, A, w, mu_as, sigma_as) # utility

# Sweep ranges
beta_min, beta_max = 0.4, 4.0
G = 10 # density of grid samples
grid_betas = [(b2, b3) for b2 in np.linspace(beta_min, beta_max, G)
                        for b3 in np.linspace(beta_min, beta_max, G)]

# Multi-start + BA controls
k0 = 8           # fresh restarts
k_warm = 4        # warm restarts
maxiter = 100
tol = 1e-9
perturb_std = 1e-3 # how much we purturb the warm starts by

# Flush partial CSV every N grid points
flush_every = 20 # if sweep is interrupted you wont lose everything

print("CWD:", os.getcwd())
print("Writing CSV to:", out_csv.resolve())

# ===================== Main sweep =====================

records = []
best_state_at = {}  # for warm starts

try:
    pbar = tqdm(total=len(grid_betas), desc="Sweeping (β2, β3)")
    for idx, (b2, b3) in enumerate(grid_betas):
        # Gather warm-start candidates
        warm_pool = []
        for j in neighbors(idx): # 
            key = grid_betas[j]
            if key in best_state_at:
                res_state = best_state_at[key]
                warm_pool.append((res_state["pogw"], res_state["pagow"]))

        stats, best_light = run_one_parallel(
            b2, b3, warm_inits=warm_pool, n_fresh=k0, n_warm=k_warm,
            max_workers=None, max_extra=12
        )

        records.append(dict(beta2=float(b2), beta3=float(b3), **stats))

        # Save posteriors ONLY if we ended up using a valid pool
        if stats.get("used_valid", False) and best_light is not None:
            fname = f"b2={b2:.6f}_b3={b3:.6f}"
            pagw = marginalizeo(best_light["pogw"], best_light["pagow"])
            np.save(POST_DIR / f"pagw_{fname}.npy",  pagw)
            np.save(POST_DIR / f"pogw_{fname}.npy",  best_light["pogw"])
            np.save(POST_DIR / f"pagow_{fname}.npy", best_light["pagow"])

            # cache best state for future warm starts
            best_state_at[(b2, b3)] = dict(
                pogw=best_light["pogw"].copy(),
                pagow=best_light["pagow"].copy(),
            )

        # Periodic flush
        if (idx + 1) % flush_every == 0:
            pd.DataFrame(records).to_csv(out_csv, index=False)

        pbar.update(1)
    pbar.close()

    df = pd.DataFrame(records).sort_values(['beta2', 'beta3']).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv.resolve()}")

except KeyboardInterrupt:
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"\nInterrupted. Partial results saved to: {out_csv.resolve()}")
    raise

# Optional quick plot
plot_J_landscape(out_csv, kind="surface",
                 save_paths=True,
                 fig_path=OUT_DIR / "J_landscape_fig.pkl",
                 png_path=OUT_DIR / "J_landscape.png")
