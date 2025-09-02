import numpy as np
import math
import matplotlib.pyplot as plt

# ---------- Entropy utilities ----------
def H10_of_t(t):
    if not (1/3 <= t <= 1):
        raise ValueError("t must be in [1/3, 1]")
    p = [t, (1-t)/2, (1-t)/2]
    return -sum(pi*math.log10(pi) for pi in p if pi > 0)

def solve_t_for_entropy(h_target, tol=1e-12, max_iter=100):
    h_min, h_max = 0.0, math.log10(3.0)
    if abs(h_target - h_min) <= tol: return 1.0
    if abs(h_target - h_max) <= tol: return 1.0/3.0
    lo, hi = 1/3, 1
    for _ in range(max_iter):
        mid = (lo + hi)/2
        h_mid = H10_of_t(mid)
        if abs(h_mid - h_target) <= tol:
            return mid
        if h_mid > h_target: lo = mid
        else: hi = mid
    return (lo+hi)/2

# ---------- Sigmoid spacing ----------
def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_spacing(u, k=1.0):
    """
    Map u in [0,1] -> v in [0,1] using a logistic with slope k at center.
    """
    lo = sigmoid(-k/2)
    hi = sigmoid(k/2)
    return (sigmoid(k*(u-0.5)) - lo) / (hi - lo)

# ---------- Generate ----------
num = 10
H_max = math.log10(3.0)
u = np.linspace(0, 1, num)

v = sigmoid_spacing(u, k=5.0)  # adjust k for more/less curvature
target_entropies = v * H_max

solutions = []
for h in target_entropies:
    t = solve_t_for_entropy(float(h))
    p = np.array([t, (1 - t)/2, (1 - t)/2])
    solutions.append((float(h), t, p))

# ---------- Print ----------
print("Sigmoid-spaced entropies and distributions:")
for i, (h, t, p) in enumerate(solutions):
    print(f"{i:2d} | H10={h:.6f} | t={t:.6f} | p={p}")

# ---------- Plot ----------
x_axis = np.linspace(0, 10, num)
entropies = [h for h,_,_ in solutions]

plt.figure(figsize=(6,4))
plt.plot(x_axis, entropies, marker='o')
plt.xlabel("Index (0..10)")
plt.ylabel("Entropy H_10")
plt.title("Entropy vs index (sigmoid spacing)")
plt.tight_layout()

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.ravel()
states = np.arange(1,4)
for i, (h, t, p) in enumerate(solutions):
    ax = axes[i]
    ax.bar(states, p)
    ax.set_xticks(states)
    ax.set_ylim(0,1.05)
    ax.set_title(f"H10={h:.3f}")
plt.tight_layout()
plt.show()
