import numpy as np
from scipy.stats import norm

def comp(p, q):
    # Avoid division by zero and log(0) by masking out zero entries in p
    mask = (p < 1) 
    return float(np.sum(p[mask] * (np.log(p[mask]+1e-12) - np.log(q[mask] + 1e-12)))) # pk changed

def acc(q, p_x_s):
    return float(np.dot(q, np.log(p_x_s + 1e-12)))

def compute_posterior(x, mus, sigmas):
    likelihoods = np.array([norm.pdf(x, loc=mu, scale=sigma) for mu, sigma in zip(mus, sigmas)])
    posterior = likelihoods / np.sum(likelihoods)
    return posterior, likelihoods

def compute_pairwise_posterior(x, mus, sigmas, idx1, idx2):
    #############PK changed so that is still 3 choices, just one has 0 posterior and likelihood
    likelihoods = np.array([
        norm.pdf(x, loc=mus[idx1], scale=sigmas[idx1]),
        norm.pdf(x, loc=mus[idx2], scale=sigmas[idx2]),
        0
    ])
    posterior = likelihoods / np.sum(likelihoods)
    return posterior, likelihoods

def compute_pairwise_posterior(x, mus, sigmas, idx1, idx2):
    likelihoods = np.array([
        norm.pdf(x, loc=mus[idx1], scale=sigmas[idx1]),
        norm.pdf(x, loc=mus[idx2], scale=sigmas[idx2])
    ])
    posterior = likelihoods / np.sum(likelihoods)
    return posterior, likelihoods




