import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from itertools import combinations

def generate_p_s_phis(k):
    """
    Generate all possible n-hot binary vectors for n = 1 to k, each of length k.
    
    Returns:
        dict[int, np.ndarray]: Dictionary where key is n (number of ones), 
                               value is array of shape (num_vectors, k).
    """
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
    """
    Performs structural inference over φ and selects MAP variable.
    
    φ* = argmax_φ Σ_s p(x|s,φ)p(s|φ)
    
    Args:
        x: np.array shape (1,) observed stimulus
        p_s_given_phi: np.array shape (num_phi, num_categories) structural likelihood matrix
        mus: np.array shape (num_categories,) means of Gaussians for each category
        sigmas: np.array shape (num_categories,) std dev of each category

    Returns:
        phi_star: np.array binary vector φ* of shape (num_categories,)
    """
    
    # calculate the data likelihood for individual Gaussian likelihoods
    likelihoods = norm.pdf(x, loc=mus, scale=sigmas)  # shape (3,)
    
    # computes the dot product of each structure defined in a row of p_s_phi with the likelihoods
    p_phi_x = p_s_phi @ likelihoods
    
    # calculate the argmax of the structural posterior
    map_idx = np.argmax(p_phi_x)
    
    # return phi_star (best model structure)
    phi_star = p_s_phi[map_idx]
    
    return phi_star, map_idx
    
    
def perceptual_inference(x, phi_star, mus, sigmas):
    """
    Conditional perceptual inference to find the posterior probability of the 
    (reduced number of) latent states.
    
    p(s|x,φ*) ∝ p(x|s,φ*)p(s|φ*)
    
    """
    
    likelihoods = norm.pdf(x, loc=mus, scale=sigmas)
        
    # use φ* as a binary mask to pick out relevant Gaussians
    reduced_likelihoods = likelihoods[phi_star.astype(bool)]
    
    # normalise Gaussian likelihoods to get posterior
    posterior = reduced_likelihoods / np.sum(reduced_likelihoods)
    
    map_idx = np.argmax(posterior)
    
    return posterior, map_idx
    
def complexity(q, p):
    """
    Complexity is the KL-divergence between the posterior q and prior p
    
    Σ_s Q(s) ln (q(s) / p(s))
    """
    
    eps = 1e-9
    # check valid probability distributions with tolerance
    if not (np.isclose(q.sum(), 1, atol=eps) and np.isclose(p.sum(), 1, atol=eps)):
        raise ValueError("q and p not valid probabilities. Each must sum to 1.")
    
    # avoid log(0) issues
    q = np.clip(q, eps, 1)
    p = np.clip(p, eps, 1)
    
    ln_q_p = np.log(q/p)
    complexity = np.dot(q, ln_q_p)
    
    return float(complexity)

def accuracy(q, p_x_s):
    """
    Accuracy is the expected log likelihood of the data given the posterior q
    """
    eps = 1e-9 # avoid log(0)
    acc = np.dot(q, np.log(p_x_s + eps))
    return float(acc)



    
def main():
    rng = np.random.default_rng(0)
    # 1D case
    
    # number of latent states
    k = 3
    reduced_k = 2
    
    # generate all structural likelihoods p(s|φ) for k latents
    p_s_phis = generate_p_s_phis(k)
    
    # pick out a single structural likelihood to use
    p_s_phi = p_s_phis[reduced_k]
    
    prior_full = np.ones(k,) /k # flat prior over full state posterior    
    prior_red = np.ones(reduced_k,) /reduced_k
    
    full_posteriors = []
    reduced_posteriors = []
    Fs_full = [] # Free energies
    Cs_full = [] # Complexity
    As_full = [] # Accuracy
    
    Fs_red = [] # Free energies
    Cs_red = [] # Complexity
    As_red = [] # Accuracy
    
    mus = np.array([0,1,2]) # means of likelihoods, shape (k,)
    sigmas = np.array([1,1,1]) # stds of likelihoods, shape (k,)
    
    # true data distribution (shifting mean along horizontal axis)
    true_mus = np.linspace(-1, 3, 50)  # horizontal axis values
    sigma_true = 0.3                    # true observation noise
    batch = 100                         # samples per true mean

    # storage for (batch-averaged) metrics vs true mean
    Fs_full, Cs_full, As_full = [], [], []
    Fs_red,  Cs_red,  As_red  = [], [], []

    for true_mu in true_mus:
        # sample a batch from the true distribution N(true_mu, sigma_true^2)
        xs_batch = rng.normal(loc=true_mu, scale=sigma_true, size=batch)

        # accumulate metrics over the batch
        C_full_sum = A_full_sum = F_full_sum = 0.0
        C_red_sum  = A_red_sum  = F_red_sum  = 0.0

        for x in xs_batch:
            likelihoods = norm.pdf(x, loc=mus, scale=sigmas)

            # structural inference
            phi_star, _ = structural_inference(x, p_s_phi, mus, sigmas)

            # full-state inference
            post_full, _ = perceptual_inference(x, np.array([1, 1, 1]), mus, sigmas)
            c_full = complexity(post_full, prior_full)
            a_full = accuracy(post_full, likelihoods)
            f_full = a_full - c_full

            # reduced-state (conditional on φ*)
            reduced_likelihoods = likelihoods[phi_star.astype(bool)]
            post_red, _ = perceptual_inference(x, phi_star, mus, sigmas)
            c_red = complexity(post_red, prior_red)
            a_red = accuracy(post_red, reduced_likelihoods)
            f_red = a_red - c_red

            C_full_sum += c_full; A_full_sum += a_full; F_full_sum += f_full
            C_red_sum  += c_red;  A_red_sum  += a_red;  F_red_sum  += f_red

        # average over the batch
        Cs_full.append(C_full_sum / batch); As_full.append(A_full_sum / batch); Fs_full.append(F_full_sum / batch)
        Cs_red.append(C_red_sum / batch);   As_red.append(A_red_sum / batch);   Fs_red.append(F_red_sum / batch)

    # plots vs the true means
    plt.plot(true_mus, Cs_full, label="Complexity (full)", color = "blue")
    plt.plot(true_mus, Cs_red, label="Complexity (reduced)", color = "dodgerblue")     
    plt.plot(true_mus, As_full, label="Accuracy (full)", color = "green")
    plt.plot(true_mus, As_red, label="Accuracy (reduced)", color = "limegreen")
    plt.title("F of full vs 2-state posterior")
    plt.xlabel("Horizontal target location")
    plt.ylabel("Complexity / accuracy")
    plt.legend()
    plt.show()

    plt.plot(true_mus, Fs_full, label="F (full)", color = "orange")
    plt.plot(true_mus, Fs_red, label="F (reduced)", color = "gold")
    plt.title("F of full vs 2-state posterior")
    plt.xlabel("Horizontal target location")
    plt.ylabel("Free Energy F")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()  
    
    
