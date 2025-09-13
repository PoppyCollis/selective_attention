import numpy as np

def max_expected_utility(mu_as, sigma_as, L, Uhi, num_samples=100000):
    """
    Compute maximum expected utility for Gaussians with same sigma
    and uniform p(x) over [L, Uhi].
    
    Parameters
    ----------
    mu_as : array (A,)
        Means of Gaussian likelihoods.
    sigma_as : array (A,)
        Standard deviations for each Gaussian (must all be equal here).
    L, Uhi : floats
        Lower and upper bounds of uniform support for x.
    num_samples : int
        Number of uniform samples for Monte Carlo integration.
    """
    d = 1  # 1D support
    sigma = sigma_as[0]
    
    # sample uniformly from support
    xs = np.random.uniform(L, Uhi, size=num_samples)
    
    # compute squared distance to nearest mean
    d2 = np.min((xs[:, None] - mu_as[None, :])**2, axis=1)
    
    # average min squared distance
    mean_min_dist2 = np.mean(d2)
    
    # constant term from Gaussian log-likelihood
    const = -0.5 * d * np.log(2 * np.pi * sigma**2)
    
    # max expected utility
    Umax = const - (1.0 / (2 * sigma**2)) * mean_min_dist2
    return Umax

# Example usage
mu_as = np.array([-96, -59, 96])
A = len(mu_as)
sigma_as = np.ones(A) * 64
epsilon1 = 64
epsilon2 = 64
L, Uhi = mu_as[0] - epsilon1, mu_as[-1] + epsilon2

Umax = max_expected_utility(mu_as, sigma_as, L, Uhi, num_samples=200000)
print("Maximum expected utility =", Umax)
