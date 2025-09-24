# minimal_eta_plot.py

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D)

# ---- config you can tweak ----
IN_CSV   = Path("results/sweep_betas_results.csv")
BACKUP   = Path("results/sweep_betas_results.backup.csv")
OUT_CSV  = Path("results/sweep_betas_results_with_eta.csv")
SCALE    = 100.0  # set your scalar here (eta = EU_best / SCALE)
# --------------------------------

# 1) backup the original CSV
BACKUP.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(IN_CSV, BACKUP)
print(f"Backup written to: {BACKUP.resolve()}")

# 2) load, add 'eta', save modified CSV
df = pd.read_csv(IN_CSV)

# tolerate alternative column names beta_2/beta_3
if "beta_2" in df.columns and "beta2" not in df.columns:
    df = df.rename(columns={"beta_2": "beta2"})
if "beta_3" in df.columns and "beta3" not in df.columns:
    df = df.rename(columns={"beta_3": "beta3"})

required = {"beta2", "beta3", "EU_best"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"CSV missing required columns: {missing}")

if SCALE == 0.0:
    raise ValueError("SCALE must be non-zero")

#df["eta"] = (1/ df["EU_best"]) / (1/-5.200799958) J_best
df["eta"] = df["J_best"]

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"Modified CSV written to: {OUT_CSV.resolve()}")

# 3) plot 3D surface of (beta2, beta3, eta)
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

def try_rectangular_surface():
    # pivot assumes a rectangular grid; NaNs indicate missing cells
    pivot = df.pivot_table(index="beta3", columns="beta2", values="eta", aggfunc="mean")
    if pivot.isna().values.any():
        return None
    X = pivot.columns.values
    Y = pivot.index.values
    Xg, Yg = np.meshgrid(X, Y)
    Z = pivot.values
    return ax.plot_surface(Xg, Yg, Z, cmap="viridis", edgecolor="none", antialiased=True)

surf = try_rectangular_surface()
if surf is None:
    # irregular/path data: triangulate
    x = df["beta2"].to_numpy()
    y = df["beta3"].to_numpy()
    z = df["eta"].to_numpy()
    tri = mtri.Triangulation(x, y)
    surf = ax.plot_trisurf(tri, z, cmap="viridis", linewidth=0.0, antialiased=True)

ax.set_xlabel(r"$\beta_2$")
ax.set_ylabel(r"$\beta_3$")
ax.set_zlabel(r"$\eta = \mathrm{EU}_{\mathrm{best}}/\mathrm{SCALE}$")
ax.view_init(elev=30, azim=45)
fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, label=r"$\eta$")
plt.tight_layout()
plt.show()
