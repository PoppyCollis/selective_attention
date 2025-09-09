import numpy as np
from scipy.stats import norm


def comp(p, q):
    # Avoid division by zero and log(0) by masking out zero entries in p
    mask = (p < 1) 
    return float(np.sum(p[mask] * (np.log(p[mask]+1e-12) - np.log(q[mask] + 1e-12)))) # pk changed

def acc(q, p_x_s):
    return float(np.dot(q, np.log(p_x_s)))

def compute_posterior(x, mus, sigmas):
    likelihoods = np.array([norm.pdf(x, loc=mu, scale=sigma) for mu, sigma in zip(mus, sigmas)])
    posterior = likelihoods / np.sum(likelihoods)
    return posterior, likelihoods

# def compute_pairwise_posterior(x, mus, sigmas, idx1, idx2):
#     #############PK changed so that is still 3 choices, just one has 0 posterior and likelihood
#     likelihoods = np.array([
#         norm.pdf(x, loc=mus[idx1], scale=sigmas[idx1]),
#         norm.pdf(x, loc=mus[idx2], scale=sigmas[idx2]),
#         0
#     ])
#     posterior = likelihoods / np.sum(likelihoods)
#     return posterior, likelihoods

def compute_pairwise_posterior(x, mus, sigmas, idx1, idx2):
    likelihoods = np.array([
        norm.pdf(x, loc=mus[idx1], scale=sigmas[idx1]),
        norm.pdf(x, loc=mus[idx2], scale=sigmas[idx2])
    ])
    posterior = likelihoods / np.sum(likelihoods)
    return posterior, likelihoods

def compute_degenerate_posterior(x, mus, sigmas, idx):
    likelihood = np.array([norm.pdf(x, loc=mus[idx], scale=sigmas[idx])])    
    posterior = likelihood
    return posterior, likelihood




x = -5.0
mus = [0.0, 1.0, 2.0]
sigmas = [1.0, 1.0, 1.0]

posterior, lik = compute_posterior(x, mus, sigmas)
print("Full posterior: ", posterior)  # e.g., [0.106, 0.352, 0.542]

posterior2, lik2 = compute_pairwise_posterior(x, mus, sigmas, 0, 1)
print("Reduced posterior 0, 1: ", posterior2)  # e.g., [0.163, 0.837]

posterior3, lik3 = compute_pairwise_posterior(x, mus, sigmas, 0, 2)
print("Reduced posterior 0, 2: ", posterior3)  # e.g., [0.163, 0.837]

posterior4, lik4 = compute_pairwise_posterior(x, mus, sigmas, 1, 2)
print("Reduced posterior 1, 2: ", posterior4)  # e.g., [0.163, 0.837]

# uninformative priors
p2 = np.array([1/2,1/2, 0]) ############PK changed
p3 = np.array([1/3,1/3,1/3])

# degenerate case
x = 1.0
mus = [0.0, 1.0, 2.0]
sigmas = [1.0, 1.0, 1.0]

posterior, lik = compute_posterior(x, mus, sigmas)
print("Full posterior: ", posterior)  # e.g., [0.106, 0.352, 0.542]

posterior2, lik2 = compute_pairwise_posterior(x, mus, sigmas, 0, 1)
print("Reduced posterior 0, 1: ", posterior2)  # e.g., [0.163, 0.837]

posterior3, lik3 = compute_pairwise_posterior(x, mus, sigmas, 0, 2)
print("Reduced posterior 0, 2: ", posterior3)  # e.g., [0.163, 0.837]

posterior4, lik4 = compute_pairwise_posterior(x, mus, sigmas, 1, 2)
print("Reduced posterior 1, 2: ", posterior4)  # e.g., [0.163, 0.837]

pos1, l1 = compute_degenerate_posterior(x, mus, sigmas, idx=0)
pos2, l2 = compute_degenerate_posterior(x, mus, sigmas, idx=1)
pos3, l3 = compute_degenerate_posterior(x, mus, sigmas, idx=2)

a1 = acc(pos1, l1)
a2 = acc(pos2, l2)
a3 = acc(pos3, l3)
a_s = []

a_s.add(a1,a2,a3)

p4 = np.array([1.0])
c1  = comp(pos1, p4)
c2  = comp(pos2, p4)
c3  = comp(pos3, p4)

c_s = []
c_s.add(c1,c2,c3)

print("degenerate cases")
print(a_s)
print(c_s)



# uninformative priors
p2 = np.array([1/2,1/2]) 
p3 = np.array([1/3,1/3,1/3])

# # posterior
# q3 = np.array([0.7,0.2,0.1])
# # renormalised 2 state posterior
# q2 = np.array([(0.7/0.9),(0.2/0.9)]) 

# complexity3 = comp(p3, q3)
# complexity2 = comp(p2, q2)
# print(complexity3,complexity2)


acc1 = acc(posterior, lik)
acc2 = acc(posterior2, lik2)
acc3 = acc(posterior3, lik3)
acc4 = acc(posterior4, lik4)
accuracies = [acc1,acc2,acc3,acc4]


comp1 = comp(posterior, p3)
comp2 = comp(posterior2, p2)
comp3 = comp(posterior3, p2)
comp4 = comp(posterior4, p2)
complexities = [comp1,comp2,comp3,comp4]


Fs = np.array(complexities) - np.array(accuracies)

print(accuracies)
print(complexities)

print(Fs)

"""
High accuracy and low complexity leads to larger values of F though...! 

Here we have that the reduced model is lower in complexity and roughly the same in accuracy...

seem to be getting higher accuracy AND reduced complexity which is promising.

Seems to be that 2 state is pretty much always betterAcc. What about 1 state though...


Now for the Li and Ma data.

1. Run the 2 state and 3 state attention models on large batch of target locations for different configs.
2. Plot the F = complexity - accuracy for each of the models per target location.
3. For each target location, ientify the model with the highest F

My question: lets say we have a point in between three equally spaced 3D gaussians (config 1 or 2 in exp 2)
Picking the two state posterior will reduce the complexity, AND the accuracy because the accuracy measure does 
not seem to take into acount any "true" s which is weird.. when we do E_Q[ln p(x|s)] we are looking at the expected 
log likelihood of the data - not the actual true state. Reducing to a two state posterior coul lead you to get it
very wrong as you might end up selecting for posteriors that arent even in the running

But then - why would this be an issue. If you faithfully represented the 3 state posterior, taking the argmax is a
random decision rule, and the same if you reduced the posterior to 2 (randomly) and then (randomly) decided on 1 or the
other.

"""