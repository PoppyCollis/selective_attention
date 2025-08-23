import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from itertools import combinations
import matplotlib.colors as mcolors

# ---------- helpers ----------
def generate_p_s_phis(k):
    result = {}
    for n in range(1, k + 1):
        rows = []
        for ones_indices in combinations(range(k), n):
            vec = np.zeros(k, dtype=int)
            vec[list(ones_indices)] = 1
            rows.append(vec)
        result[n] = np.array(rows)
    return result

def structural_inference(x, p_s_phi, mus, sigmas):
    likelihoods = norm.pdf(x, loc=mus, scale=sigmas)  # shape (k,)
    p_phi_x = p_s_phi @ likelihoods
    map_idx = np.argmax(p_phi_x)
    phi_star = p_s_phi[map_idx]
    return phi_star, map_idx

def perceptual_inference(x, phi_star, mus, sigmas):
    likelihoods = norm.pdf(x, loc=mus, scale=sigmas)
    reduced_likelihoods = likelihoods[phi_star.astype(bool)]
    posterior = reduced_likelihoods / np.sum(reduced_likelihoods)
    map_idx = np.argmax(posterior)
    return posterior, map_idx

def complexity(q, p):
    eps = 1e-9
    if not (np.isclose(q.sum(), 1, atol=eps) and np.isclose(p.sum(), 1, atol=eps)):
        raise ValueError("q and p not valid probabilities. Each must sum to 1.")
    q = np.clip(q, eps, 1); p = np.clip(p, eps, 1)
    return float(np.dot(q, np.log(q / p)))

def accuracy(q, p_x_s):
    eps = 1e-9
    return float(np.dot(q, np.log(p_x_s + eps)))

def lighten_color(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except KeyError:
        c = color
    c = mcolors.to_rgb(c)
    white = np.array([1, 1, 1])
    return tuple((1 - amount) * np.array(c) + amount * white)

# ---------- main experiment ----------
def main():
    rng = np.random.default_rng(0)

    # latent states
    k = 3
    reduced_k = 2

    # structures: pick 2-hot family for structural inference
    p_s_phis = generate_p_s_phis(k)
    p_s_phi = p_s_phis[reduced_k]

    prior_full = np.ones(k) / k
    prior_red  = np.ones(reduced_k) / reduced_k

    # internal model (fixed)
    model_mus = np.array([0., 1., 2.])
    model_sigmas = np.array([1., 1., 1.])

    # true data: 3-state mixture with equal weights; means shift together
    base_true_mus = np.array([0., 1., 2.])
    num_steps = 60
    shifts = np.linspace(-1.0, 1.0, num_steps)   # shift range; adjust as needed
    sigma_true = 0.3
    batch = 200

    # storage
    Fs_full, Cs_full, As_full = [], [], []
    Fs_red,  Cs_red,  As_red  = [], [], []

    # uniform component weights (always 3-state, equal weight)
    mix_weights = np.ones(3) / 3

    for shift in shifts:
        true_mus = base_true_mus + shift  # shift all three components equally

        # sample batch from the mixture N(true_mus[c], sigma_true^2)
        comps = rng.choice(3, size=batch, p=mix_weights)
        xs_batch = rng.normal(loc=true_mus[comps], scale=sigma_true, size=batch)

        C_full_sum = A_full_sum = F_full_sum = 0.0
        C_red_sum  = A_red_sum  = F_red_sum  = 0.0

        for x in xs_batch:
            likelihoods = norm.pdf(x, loc=model_mus, scale=model_sigmas)

            # structural inference (over 2-hot structures)
            phi_star, _ = structural_inference(x, p_s_phi, model_mus, model_sigmas)

            # full-state inference
            post_full, _ = perceptual_inference(x, np.array([1, 1, 1]), model_mus, model_sigmas)
            c_full = complexity(post_full, prior_full)
            a_full = accuracy(post_full, likelihoods)
            f_full = a_full - c_full

            # reduced (conditional on φ*)
            reduced_lk = likelihoods[phi_star.astype(bool)]
            post_red, _ = perceptual_inference(x, phi_star, model_mus, model_sigmas)
            c_red = complexity(post_red, prior_red)
            a_red = accuracy(post_red, reduced_lk)
            f_red = a_red - c_red

            C_full_sum += c_full; A_full_sum += a_full; F_full_sum += f_full
            C_red_sum  += c_red;  A_red_sum  += a_red;  F_red_sum  += f_red

        # batch averages
        Cs_full.append(C_full_sum / batch); As_full.append(A_full_sum / batch); Fs_full.append(F_full_sum / batch)
        Cs_red.append(C_red_sum / batch);   As_red.append(A_red_sum / batch);   Fs_red.append(F_red_sum / batch)

    # ---------- plots (reduced = lighter versions of full) ----------
    x_axis = shifts  # x-axis is the horizontal shift

    color_C = "blue"
    color_A = "green"
    color_F = "orange"
    color_C_light = lighten_color(color_C, 0.5)
    color_A_light = lighten_color(color_A, 0.5)
    color_F_light = lighten_color(color_F, 0.5)

    plt.plot(x_axis, Cs_full, label="Complexity (full)", color=color_C)
    plt.plot(x_axis, Cs_red,  label="Complexity (reduced)", color=color_C_light)
    plt.plot(x_axis, As_full, label="Accuracy (full)", color=color_A)
    plt.plot(x_axis, As_red,  label="Accuracy (reduced)", color=color_A_light)
    plt.title("Complexity & Accuracy vs horizontal shift of true mixture")
    plt.xlabel("Shift applied to true means")
    plt.ylabel("Complexity / Accuracy")
    plt.legend()
    plt.show()

    plt.plot(x_axis, Fs_full, label="F (full)", color=color_F)
    plt.plot(x_axis, Fs_red,  label="F (reduced)", color=color_F_light)
    plt.title("Free Energy vs horizontal shift of true mixture")
    plt.xlabel("Shift applied to true means")
    plt.ylabel("Free Energy F")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
