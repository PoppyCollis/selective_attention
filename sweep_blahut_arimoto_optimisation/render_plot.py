import pickle
import matplotlib.pyplot as plt

with open("J_landscape_fig_config3.pkl", "rb") as f:
    fig = pickle.load(f)

# The figure is live: rotate, change view, or update axes
ax = fig.axes[0]
ax.view_init(elev=45, azim=120)
plt.show()

# render_plot.py
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
#!/usr/bin/env python3
# plot_eu_surface.py
import argparse
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D)
import matplotlib.tri as mtri


def plot_eu_surface(csv_path: str,
                    value_col: str = "EU_best",
                    elev: float = 30,
                    azim: float = 45,
                    cmap: str = "viridis",
                    figsize=(9, 7),
                    prefer_triangulation: bool = False,
                    logx: bool = False,
                    logy: bool = False,
                    save: bool = False,
                    png_path: str = "EU_surface.png",
                    fig_path: str = "EU_surface.pkl"):
    """
    Plot a 3D surface of `value_col` (default: EU_best) over (beta2, beta3).

    - Uses a regular surface if the (beta2,beta3) pairs form a rectangle.
    - Falls back to a triangulated surface for irregular/path data.
    - Optionally saves PNG and a pickled Matplotlib figure.

    Returns (fig, ax).
    """
    df = pd.read_csv(csv_path)
    required = {"beta2", "beta3", value_col}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")

    # Keep last occurrence per (beta2,beta3) if duplicates exist
    df = df.dropna(subset=list(required)).drop_duplicates(
        subset=["beta2", "beta3"], keep="last"
    )

    # Prepare labels (support log axes)
    x_label = r"$\log_{10}\,\beta_2$" if logx else r"$\beta_2$"
    y_label = r"$\log_{10}\,\beta_3$" if logy else r"$\beta_3$"

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    def try_rectangular_surface():
        # Pivot assumes grid; check for missing cells
        pivot = df.pivot_table(index="beta3", columns="beta2",
                               values=value_col, aggfunc="mean")
        if pivot.isna().values.any():
            return None
        X = pivot.columns.values
        Y = pivot.index.values
        Xg, Yg = np.meshgrid(X, Y)
        if logx:
            Xg = np.log10(Xg)
        if logy:
            Yg = np.log10(Yg)
        Z = pivot.values
        return ax.plot_surface(Xg, Yg, Z, cmap=cmap, edgecolor="none", antialiased=True)

    surf = None
    if not prefer_triangulation:
        surf = try_rectangular_surface()

    if surf is None:
        # Irregular/path data: use triangulation
        x = df["beta2"].to_numpy()
        y = df["beta3"].to_numpy()
        if logx:
            x = np.log10(x)
        if logy:
            y = np.log10(y)
        z = df[value_col].to_numpy()
        triang = mtri.Triangulation(x, y)
        surf = ax.plot_trisurf(triang, z, cmap=cmap, linewidth=0.0, antialiased=True)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(value_col)
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, label=value_col)
    plt.tight_layout()

    if save:
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=300)
        with open(fig_path, "wb") as f:
            pickle.dump(fig, f)
        print(f"Saved PNG: {Path(png_path).resolve()}")
        print(f"Saved pickled figure: {Path(fig_path).resolve()}")

    plt.show()
    return fig, ax

# 1) Resolve CSV path robustly
candidates = [
    Path("sweep_betas_results_config3.csv"),
    Path("results/sweep_betas_results_config3.csv"),
    Path("results/sweep_betas_results_paths.csv"),
]
csv_path = next((p for p in candidates if p.exists()), None)
if csv_path is None:
    raise FileNotFoundError("Could not find a sweep_betas_results*.csv in ./ or ./results/")

# 2) Decide where to save artifacts (ensure dir exists)
fig_dir = Path("results/figs")
fig_dir.mkdir(parents=True, exist_ok=True)
fig_pkl = fig_dir / "J_landscape_fig_config3.pkl"
fig_png = fig_dir / "J_landscape_config3.png"

# 3) Make & save the plot (set save=True and pass the SAME fig_path you’ll load)
plot_eu_surface(
    csv_path=str(csv_path),
    value_col="EU_best",
    elev=30, azim=45,
    cmap="viridis",
    prefer_triangulation=False,  # set True if data aren’t on a grid
    logx=False, logy=False,
    save=True,
    png_path=str(fig_png),
    fig_path=str(fig_pkl),
)

print(f"Saved pickle to: {fig_pkl.resolve()}")

# 4) Reload and re-render
with open(fig_pkl, "rb") as f:
    fig = pickle.load(f)

ax = fig.axes[0]
ax.view_init(elev=45, azim=120)  # adjust camera
plt.show()

