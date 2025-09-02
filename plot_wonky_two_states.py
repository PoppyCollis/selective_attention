import numpy as np
import matplotlib.pyplot as plt

def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

one_state = normalize(np.load("negent.npy"))
two_state_inf = normalize(np.load("negent_2.npy"))
two_state_1 = normalize(np.load("negent_3.npy"))
diff_1 = normalize(np.load("diff.npy"))
max_1 = normalize(np.load("max.npy"))


xs = np.linspace(0, one_state.shape[0], one_state.shape[0])

plt.plot(xs, one_state, label="|φ|=1")
plt.plot(xs, diff_1, label="Diff")
plt.plot(xs, max_1, label="Max")
plt.plot(xs, two_state_inf, label="|φ|=1, α = ∞")
plt.plot(xs, two_state_1, label="|φ|=1, α = 1")
plt.legend()
plt.show()