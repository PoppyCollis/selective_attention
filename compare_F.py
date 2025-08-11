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
    
    print(likelihoods)
    
    # use φ* as a binary mask to pick out relevant Gaussians
    reduced_likelihoods = likelihoods[phi_star.astype(bool)]
    
    # normalise Gaussian likelihoods to get posterior
    posterior = reduced_likelihoods / np.sum(reduced_likelihoods)
    
    map_idx = np.argmax(posterior)
    
    return posterior, map_idx
    
    
    
def main():
    
    # 1D case
    
    # number of latent states
    k = 3
    
    # generate all structural likelihoods p(s|φ) for k latents
    p_s_phis = generate_p_s_phis(k)
    
    # pick out a single structural likelihood to use
    p_s_phi = p_s_phis[2]
    
    
    # x = 0.5 # observation
    xs = np.linspace(-1,3,10)
    mus = np.array([0,1,2]) # means of likelihoods, shape (k,)
    sigmas = np.array([1,1,1]) # std of likelihoods, shape (k,)
    
    for x in xs:
        phi_star, map_idx_phi = structural_inference(x, p_s_phi, mus, sigmas)
        print(phi_star)
        
        post, map_idx_s = perceptual_inference(x, phi_star, mus, sigmas)
        print(post)
    

if __name__ == "__main__":
    main()



# def structural_inference():
#     pass




# def comp(p, q):
#     # Avoid division by zero and log(0) by masking out zero entries in p
#     mask = (p < 1) 
#     return float(np.sum(p[mask] * (np.log(p[mask]+1e-12) - np.log(q[mask] + 1e-12)))) # pk changed

# def acc(q, p_x_s):
#     return float(np.dot(q, np.log(p_x_s + 1e-12)))

# def compute_posterior(x, mus, sigmas):
#     likelihoods = np.array([norm.pdf(x, loc=mu, scale=sigma) for mu, sigma in zip(mus, sigmas)])
#     posterior = likelihoods / np.sum(likelihoods)
#     return posterior, likelihoods

# def compute_pairwise_posterior(x, mus, sigmas, idx1, idx2):
#     #############PK changed so that is still 3 choices, just one has 0 posterior and likelihood
#     likelihoods = np.array([
#         norm.pdf(x, loc=mus[idx1], scale=sigmas[idx1]),
#         norm.pdf(x, loc=mus[idx2], scale=sigmas[idx2]),
#         0
#     ])
#     posterior = likelihoods / np.sum(likelihoods)
#     return posterior, likelihoods

# def compute_pairwise_posterior(x, mus, sigmas, idx1, idx2):
#     likelihoods = np.array([
#         norm.pdf(x, loc=mus[idx1], scale=sigmas[idx1]),
#         norm.pdf(x, loc=mus[idx2], scale=sigmas[idx2])
#     ])
#     posterior = likelihoods / np.sum(likelihoods)
#     return posterior, likelihoods


# def structural_inference():
#     pass


# # represent p and q right way round in the KL divergence for complexity
# # check mask logic


# xs = np.linspace(-3,3, 100)
# #x = -5.0
# mus = [0.0, 1.0, 2.0]
# sigmas = [1.0, 1.0, 1.0]


# for x in xs:
#     # compute posterior full model 
#     # compute posterior reduced 2 prior 2
#     # compute posterior reduced 2, prior 3
    
#     # record F: C, A 
