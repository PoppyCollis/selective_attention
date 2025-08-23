import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Style ---
mpl.rcParams.update({
    "figure.dpi": 160,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "xtick.direction": "out",
    "ytick.direction": "out"
})

# Data for Gaussians
x = np.linspace(-4, 5, 400)
sig = 1.0
means = [0, 1, 2]

def gaussian(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mu)/sigma)**2)

fig, ax = plt.subplots(figsize=(7, 3))

colors = ["#009E73", "#CC79A7", "#56B4E9"]  # colorblind-friendly
for m, c in zip(means, colors):
    ax.plot(x, gaussian(x, m, sig), color=c, linewidth=2.2)

# Minimal ticks
ax.set_xticks([-4,-2,0,2,4])
ax.set_yticks([0,0.5])

# No grid
ax.grid(False)

# Arrow indicating target moving from -1 to 1
ax.annotate(
    "", xy=(4, 0.45), xytext=(-2, 0.45),
    arrowprops=dict(arrowstyle="->", lw=2, color="black")
)
ax.text(1, 0.48, "Target location", ha="center", va="bottom", fontsize=12)

# Axis limits
ax.set_xlim(-3, 5)
ax.set_ylim(0, None)

plt.xlabel("x")
plt.ylabel("Probability density")
plt.tight_layout()
plt.show()


