import numpy as np
from scipy.stats import multivariate_normal

def structural_inference(x, p_s_given_phi, mus, sigmas, sigma_obs):
    """
    Performs MAP structural inference over φ.

    Args:
        x: np.array shape (2,) observed stimulus
        p_s_given_phi: shape (num_phi, num_categories) binary gating matrix
        mus: shape (num_categories, 2) means of Gaussians for each category
        sigmas: shape (num_categories,) std dev of each category
        sigma_obs: float sensory noise std dev

    Returns:
        phi_star_index: int index of best φ
        phi_star: np.array binary vector φ* of shape (num_categories,)
    """
    num_phi, num_categories = p_s_given_phi.shape
    px_given_phi = np.zeros(num_phi)

    for i in range(num_phi):
        phi = p_s_given_phi[i]
        px_given_s = np.zeros(num_categories)

        for s in range(num_categories):
            if phi[s] == 1:
                cov = (sigmas[s]**2 + sigma_obs**2) * np.eye(2)
                px_given_s[s] = multivariate_normal.pdf(x, mean=mus[s], cov=cov)

        num_active = phi.sum()
        if num_active == 0:
            continue  # degenerate φ

        p_s_given_this_phi = phi / num_active
        px_given_phi[i] = np.sum(px_given_s * p_s_given_this_phi)

    phi_star_index = np.argmax(px_given_phi)
    phi_star = p_s_given_phi[phi_star_index]

    return phi_star_index, phi_star


def perceptual_inference(x, phi_star, mus, sigmas, sigma_obs):
    """
    Performs perceptual inference given φ*.

    Args:
        x: np.array shape (2,)
        phi_star: np.array shape (num_categories,)
        mus, sigmas, sigma_obs: as above

    Returns:
        posterior_s: np.array shape (num_categories,) – p(s | x, φ*)
    """
    num_categories = len(phi_star)
    posterior_s = np.zeros(num_categories)

    active_indices = phi_star == 1
    num_active = active_indices.sum()
    if num_active == 0:
        return posterior_s  # degenerate case

    p_s_given_phi = np.zeros(num_categories)
    p_s_given_phi[active_indices] = 1 / num_active

    for s in range(num_categories):
        if phi_star[s] == 1:
            cov = (sigmas[s]**2 + sigma_obs**2) * np.eye(2)
            likelihood = multivariate_normal.pdf(x, mean=mus[s], cov=cov)
            posterior_s[s] = likelihood * p_s_given_phi[s]

    posterior_s /= posterior_s.sum()
    return posterior_s

def compute_free_energy(x, posterior_s, phi_star, mus, sigmas, sigma_obs):
    """
    Compute variational free energy F = accuracy - complexity.

    Args:
        x: np.array of shape (2,), observed stimulus
        posterior_s: np.array of shape (num_categories,), inferred posterior p(s | x, φ*)
        phi_star: np.array of shape (num_categories,), active categories in φ*
        mus, sigmas, sigma_obs: as before

    Returns:
        free_energy: float
        accuracy: float
        complexity: float
    """
    num_categories = len(posterior_s)
    log_px_given_s = np.full(num_categories, -np.inf)
    p_s = np.zeros(num_categories)

    active_indices = phi_star == 1
    num_active = np.sum(active_indices)

    if num_active == 0:
        return -np.inf, -np.inf, np.inf  # degenerate case

    # Prior over s: uniform over active states
    p_s[active_indices] = 1 / num_active

    for s in range(num_categories):
        if phi_star[s] == 1:
            cov = (sigmas[s]**2 + sigma_obs**2) * np.eye(2)
            log_px_given_s[s] = multivariate_normal.logpdf(x, mean=mus[s], cov=cov)

    # Accuracy: E_q[log p(x | s)]
    accuracy = np.sum(posterior_s * log_px_given_s)

    # Complexity: KL[Q(s) || p(s)]
    with np.errstate(divide='ignore', invalid='ignore'):
        log_qs_over_ps = np.where((posterior_s > 0) & (p_s > 0),
                                  np.log(posterior_s / p_s),
                                  0.0)
    complexity = np.sum(posterior_s * log_qs_over_ps)

    free_energy = accuracy - complexity
    return free_energy, accuracy, complexity



# Setup
mus = np.array([[-5, 0], [0, -5], [5, 0]])
sigmas = np.array([1.0, 1.0, 1.0])
sigma_obs = 0.5


p_s_given_phi_1 = np.array([
    [1,1,0],
    [1,0,1],
    [0,1,1]
])

p_s_given_phi_2 = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1]
])


# Simulate
true_s = 1
x = np.random.multivariate_normal(mus[true_s], (sigmas[true_s]**2 + sigma_obs**2) * np.eye(2))

# Inference
phi_star_index, phi_star = structural_inference(x, p_s_given_phi_2, mus, sigmas, sigma_obs)
posterior_s = perceptual_inference(x, phi_star, mus, sigmas, sigma_obs)

print(f"MAP φ index: {phi_star_index}, φ*: {phi_star}")
print(f"Posterior p(s | x, φ*): {posterior_s}")

free_energy, accuracy, complexity = compute_free_energy(x, posterior_s, phi_star, mus, sigmas, sigma_obs)
print(f"Free Energy: {free_energy:.3f}, Accuracy: {accuracy:.3f}, Complexity: {complexity:.3f}")
