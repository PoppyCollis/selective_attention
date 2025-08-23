import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from itertools import combinations
import matplotlib as mpl

# --- Style ---
mpl.rcParams.update({
    "figure.dpi": 160,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "xtick.direction": "out",
    "ytick.direction": "out"
})

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
    p_phi_x = p_s_phi  @  likelihoods
    
    p_phi_x /= np.sum(p_phi_x) # convert to posterior
    
    # calculate the argmax of the structural posterior
    map_idx = np.argmax(p_phi_x)
    
    # return phi_star (best model structure)
    phi_star = p_s_phi[map_idx]
    
    return phi_star, map_idx, p_phi_x
    
    
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

    beta = 1 # weight of perceptual complexity
    gamma = 1 # weight of structural complexity
    # 1D case
    
    # number of latent states
    k = 3
    reduced_k = 2

    # generate all structural likelihoods p(s|φ) for k latents
    p_s_phis = generate_p_s_phis(k)
    
    # pick out a single structural likelihood to use
    p_s_phi = p_s_phis[reduced_k]
    
    num_structures = p_s_phi.shape[0]
    
    prior_full = np.ones(k,) /k # flat prior over full state posterior    
    prior_red = np.ones(reduced_k,) /reduced_k
    prior_phi = np.ones(num_structures,) /num_structures # flat prior over structures
    
    full_posteriors = []
    reduced_posteriors = []
    
    Fs_full = [] # Free energies
    Cs_full = [] # Complexity
    As_full = [] # Accuracy
    
    Cs_phi = [] # Complexity for structural inference
    
    Fs_red = [] # Free energies
    Cs_red = [] # Complexity
    As_red = [] # Accuracy
    
    Fs_combined = [] # A - (complexity1 + complexity2)
        
    # x = 0.5 # observation
    xs = np.linspace(-1,3,100)
    mus = np.array([0,1,2]) # means of likelihoods, shape (k,)
    sigmas = np.array([1,1,1]) # stds of likelihoods, shape (k,)
    
    
    
    
    # ADDED FOR 1 STATE
    reduced_m = 1
    p_s_phi_1 = p_s_phis[reduced_m]
    prior_red_1 = np.ones(reduced_m,) /reduced_m
    num_structures_1 = p_s_phi_1.shape[0]
    prior_phi_1 = np.ones(num_structures_1,) /num_structures_1
    reduced_posteriors_1 = []
    
    Fs_red_1 = [] # Free energies
    Cs_red_1 = [] # Complexity
    As_red_1 = [] # Accuracy
    Cs_phi_1 = [] # Complexity for structural inferece
    Fs_combined_1 = [] # A - (complexity1 + complexity2)


    
    for x in xs:
        likelihoods = norm.pdf(x, loc=mus, scale=sigmas)
                
        # 1. full state inference 
        post_full, map_idx_s_full = perceptual_inference(x, np.array([1,1,1]), mus, sigmas)
        full_posteriors.append(post_full)
        c = complexity(post_full, prior_full)
        Cs_full.append(c)
        a = accuracy(post_full, likelihoods)
        As_full.append(a)
        Fs_full.append(a-(1/beta)*c)
        
        # 2.a) structural inference
        phi_star, map_idx_phi, p_phi_x = structural_inference(x, p_s_phi, mus, sigmas)

        # calculate complexity of structural inference
        c_phi = complexity(p_phi_x, prior_phi)
        Cs_phi.append(c_phi)
        
        # 2.b) reduced conditional state inference 
        reduced_likelihoods = likelihoods[phi_star.astype(bool)]
        post_red, map_idx_s = perceptual_inference(x, phi_star, mus, sigmas)
        reduced_posteriors.append(post_red)
        c = complexity(post_red, prior_red)
        Cs_red.append(c)
        a = accuracy(post_red, reduced_likelihoods)
        As_red.append(a)
        Fs_red.append(a-(1/beta)*c)
        
        Fs_combined.append(a - ( ((1/gamma)*c_phi) + ((1/beta)*c)   ))
        
        
        # ADDED FOR 1 STATE
        # 2.a) structural inference
        phi_star_1, map_idx_phi_1, p_phi_x_1 = structural_inference(x, p_s_phi_1, mus, sigmas)
        # calculate complexity of structural inference
        c_phi_1 = complexity(p_phi_x_1, prior_phi_1)
        Cs_phi_1.append(c_phi_1)
        
         # 2.b) reduced conditional state inference 
        reduced_likelihoods_1 = likelihoods[phi_star_1.astype(bool)]
        post_red_1, map_idx_s_1 = perceptual_inference(x, phi_star_1, mus, sigmas)
        reduced_posteriors_1.append(post_red_1)
        c_1 = complexity(post_red_1, prior_red_1)
        Cs_red_1.append(c_1)
        a_1 = accuracy(post_red_1, reduced_likelihoods_1)
        As_red_1.append(a_1)
        Fs_red_1.append(a_1-(1/beta)*c_1)
        
        Fs_combined_1.append(a_1 - ( ((1/gamma)*c_phi_1) + ((1/beta)*c_1)   ))
        
        
    combined_Cs_red = np.add(Cs_phi, Cs_red)
    
    # ADDED FOR 1 STATE 
    combined_Cs_red_1 = np.add(Cs_phi_1, Cs_red_1)
    
    
    
    
    
    # plt.plot(xs, Cs_red_1, label="Complexity (1 state sparse)", color = "red")
    # plt.plot(xs, Cs_phi_1, label="Complexity (phi 1 state)", color = "yellow")
    # plt.plot(xs, combined_Cs_red_1, label="Complexity (combined 1 state)", color = "orange")

    # plt.plot(xs, Cs_red, label="Complexity (2 state)", color = "blue")
    # plt.plot(xs, Cs_phi, label="Complexity (phi 2 state)", color = "lightblue")
    # plt.plot(xs, combined_Cs_red, label="Complexity (combined 2 state)", color = "dodgerblue")

    # plt.plot(xs, Cs_full, label="Complexity (full 3 state)", color = "pink")
                
    # plt.plot(xs, As_full, label="Accuracy (full 3)", color = "green")
    # plt.plot(xs, As_red, label="Accuracy (sparse 2)", color = "limegreen")
    # plt.plot(xs, As_red_1, label="Accuracy (sparse 1)", color = "aquamarine")

    
    # plt.xlabel("Horizontal target location")
    # plt.ylabel("Free Energy Components")
    # plt.legend()
    # plt.show()
    

    
    
    
    
    # plt.plot(xs, Fs_combined, label="F (combined red)", color = "red")
    # plt.plot(xs, Fs_full, label="F (full)", color = "orange")
    # plt.plot(xs, Fs_red, label="F (reduced)", color = "gold")
    # plt.title("F of full vs 2-state posterior")
    # plt.xlabel("Horizontal target location")
    # plt.ylabel("Free Energy F")
    

    # plt.tight_layout()

    # plt.legend()
    # plt.show()
    
    
    combined_F = np.add(Fs_red, Cs_phi)


    # ------------- Plot 2: Free Energy -------------
    fig, ax = plt.subplots(figsize=(7, 3))

    viridis_colors = plt.cm.viridis(np.linspace(0, 1, 6))

    # ax.plot(xs, Fs_combined, label="Attended Posterior 2", color=viridis_colors[0], linewidth=2)
    # ax.plot(xs, Fs_combined_1, label="Attended Posterior 1", color=viridis_colors[1], linewidth=2)
    # ax.plot(xs, Fs_full,    label="Full posterior",         color=viridis_colors[2], linewidth=2)
    # ax.plot(xs, Fs_red,     label="Sparse posterior 2",      color=viridis_colors[3], linewidth=2)
    # ax.plot(xs, Fs_red_1,     label="Sparse posterior 1",      color=viridis_colors[4], linewidth=2)
    
    ax.plot(xs, Fs_combined, label="Attended Posterior 2", linewidth=2)
    ax.plot(xs, Fs_combined_1, label="Attended Posterior 1", linewidth=2)
    ax.plot(xs, Fs_full,    label="Full posterior", linewidth=2)
    # ax.plot(xs, Fs_red,     label="Sparse posterior 2",     linewidth=2)
    # ax.plot(xs, Fs_red_1,     label="Sparse posterior 1",    linewidth=2)



    #ax.set_title("F of full vs 2-state posterior")
    ax.set_xlabel("Horizontal target location")
    ax.set_ylabel("Free Energy F")
    ax.legend()

    # Minimal ticks, no grid
    ax.set_xticks([-1, 1, 3])
    #ax.set_yticks([1,1.5,2,2.5])
    ax.grid(False)

    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()




    # Viridis colour cycle
    #viridis_colors = plt.cm.viridis(np.linspace(0, 1, 6))

    # Example data placeholders (replace with your arrays)
    # xs = np.linspace(-3, 3, 100)
    # Cs_phi, Cs_red, Cs_full, As_full, As_red, Fs_full, Fs_red = [np.random.rand(len(xs)) for _ in range(7)]

    # ------------- Plot 1: Complexity & Accuracy -------------
    # fig, ax = plt.subplots(figsize=(7, 3))

    # combined_Cs_red = np.add(Cs_phi, Cs_red)

    # #ax.plot(xs, combined_Cs_red, label="Complexity (combined)", color=viridis_colors[0], linewidth=2)
    # #ax.plot(xs, Cs_phi,           label="Complexity (phi)",     color=viridis_colors[1], linewidth=2)
    # ax.plot(xs, Cs_full,          label="Complexity (full)",    color=viridis_colors[2], linewidth=2)
    # ax.plot(xs, Cs_red,           label="Complexity (reduced)", color=viridis_colors[3], linewidth=2)
    # ax.plot(xs, As_full,          label="Accuracy (full)",      color=viridis_colors[4], linewidth=2)
    # ax.plot(xs, As_red,           label="Accuracy (reduced)",   color=viridis_colors[5], linewidth=2)

    # ax.set_title("F of full vs 2-state posterior")
    # ax.set_xlabel("Horizontal target location")
    # ax.set_ylabel("Complexity / accuracy")
    # ax.legend()

    # # Minimal ticks, no grid
    # ax.set_xticks([-1, 3])
    # ax.set_yticks([])
    # ax.grid(False)

    # plt.tight_layout()
    # plt.show()