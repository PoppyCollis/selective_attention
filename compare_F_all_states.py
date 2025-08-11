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
        
    # instead of using φ* as a binary mask to pick out relevant Gaussians, we do element-wise multiplication
    # this keeps the posterior of size k but with a zero entry instead...
    reduced_likelihoods = likelihoods * phi_star
    
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
    
    # x = 0.5 # observation
    xs = np.linspace(-1,3,10)
    mus = np.array([0,1,2]) # means of likelihoods, shape (k,)
    sigmas = np.array([1,1,1]) # stds of likelihoods, shape (k,)
    
    for x in xs:
        likelihoods = norm.pdf(x, loc=mus, scale=sigmas)

        # structral inference
        phi_star, map_idx_phi = structural_inference(x, p_s_phi, mus, sigmas)
                
        # full state inference 
        post_full, map_idx_s_full = perceptual_inference(x, np.array([1,1,1]), mus, sigmas)
        full_posteriors.append(post_full)
        c = complexity(post_full, prior_full)
        Cs_full.append(c)
        a = accuracy(post_full, likelihoods)
        As_full.append(a)
        Fs_full.append(a-c)

        # reduced conditional state inference 
        # instead of using φ* as a binary mask to pick out relevant Gaussians, we do element-wise multiplication
        # this keeps the posterior of size k but with a zero entry instead...
        reduced_likelihoods = likelihoods * phi_star
        post_red, map_idx_s = perceptual_inference(x, phi_star, mus, sigmas)
        print("reduced post", post_red)
        reduced_posteriors.append(post_red)
        c = complexity(post_red, prior_full)
        Cs_red.append(c)
        a = accuracy(post_red, reduced_likelihoods)
        As_red.append(a)
        Fs_red.append(a-c)

    
    plt.plot(xs, Cs_full, label="Complexity (full)", color = "blue")
    plt.plot(xs, Cs_red, label="Complexity (reduced)", color = "dodgerblue")
                
    plt.plot(xs, As_full, label="Accuracy (full)", color = "green")
    plt.plot(xs, As_red, label="Accuracy (reduced)", color = "limegreen")
    plt.title("F of full vs 2-state posterior")
    plt.xlabel("Horizontal target location")
    plt.ylabel("Complexity / accuracy")
    plt.legend()
    plt.show()

    plt.plot(xs, Fs_full, label="F (full)", color = "orange")
    plt.plot(xs, Fs_red, label="F (reduced)", color = "gold")
    plt.title("F of full vs 2-state posterior")
    plt.xlabel("Horizontal target location")
    plt.ylabel("Free Energy F")
    plt.legend()
    plt.show()
    
    
    
    
    
    

        
        
        
        
        
    

if __name__ == "__main__":
    main()
