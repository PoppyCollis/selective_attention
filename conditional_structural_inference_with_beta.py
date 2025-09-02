import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import itertools


def plot_posterior_perception(title, k, posterior, labels=None):
    # Custom colors for each bar
    colors = ['red', 'yellowgreen', 'dodgerblue']

    # Apply a clean style
    #plt.style.use('seaborn-white')

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot bars with individual colors
    bars = ax.bar(np.arange(k), posterior, color=colors[:k], edgecolor='black', linewidth=0.7, alpha=0.5)

    # Set axis ticks
    ax.set_xticks(np.arange(k))
    if labels is not None:
        ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_ylim(0, 1)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Optional grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)

    # Title styling
    ax.set_title(title, fontsize=14, weight='bold', pad=10)

    plt.tight_layout()


def construct_structural_variable_likelihoods(k, attended_states = 2):
    # construct P(s = 1|ϕ)
    if attended_states == 1:
        # likelihood P(s_i = 1|ϕ) as indicator function for each latent s_i
        p_s_phi = np.eye(k) 
    
    elif attended_states < k:
        # likelihood P(s_i = 1|ϕ) as n-wise indicators for each latent s_i
        combinations = list(itertools.combinations(range(k), attended_states))
        rows=[]
        for combo in combinations:
            row=np.zeros(k,dtype=int)
            row[list(combo)]=1
            rows.append(row)
        p_s_phi=np.array(rows)       
    
    elif attended_states == k:
        p_s_phi = np.ones((k,k),dtype=int) # no information in structural prior
    else:
        raise ValueError("Not implemented")
        
    return p_s_phi


def posterior_likelihood_norm(x, k, means, covariances, T):
    
    n_components = k
    responsibilities = np.zeros(n_components,)
    
    # Compute responsibilities for each data point and each Gaussian component
    for k in range(n_components):
        individual_likelihood = multivariate_normal.pdf(x, mean=means[k], cov=covariances[k]*T)
        responsibilities[k] = individual_likelihood
        
    # Normalize responsibilities
    posterior = responsibilities / np.sum(responsibilities)
    
    return posterior


def posterior_with_attention_beta(x, k, means, covariances, p_s_phi):
    
    """
    Instead of argmax of posterior over phi, we have a temperature parameter
    which effectively returns us a one hot vector posterior over phi
    which we take as a prior in the inference of p(s|x)
    
    """ 

    beta = 1 # very low temperature
    
    p_x_s = np.zeros(k,)  # list of Gaussian likelihoods P(x| s_i=1, phi)

    for i in range(k):
        p_x_s[i] = (multivariate_normal.pdf(
            x, mean=means[i], cov=covariances[i]))
        
    num_phis = p_s_phi.shape[0]

    # infer posterior P(ϕ|x) of structural latent variable
    post_phi_x = np.dot(p_s_phi, p_x_s.T)
    # normalise the posterior
    post_phi_x = post_phi_x / np.sum(post_phi_x)
    
    print(post_phi_x)

    #plot_posterior_perception('posterior P(ϕ|x) ', 3, post_phi_x)
    
    # infer posterior P(S|ϕ, x) given posterior over ϕ as prior
    post_s_x = p_x_s*post_phi_x**beta
    # post_s_x = p_x_s * p_s_phi[np.argmax(post_phi_s)]
    # normalise posterior
    post_s_x = post_s_x/np.sum(post_s_x)
    
    return post_s_x


def posterior_with_attention_argmax(x, k, means, covariances, p_s_phi):
    
    """
    This is where I take an argmax of the first inference of posterior over phi
    
    """

    p_x_s = np.zeros(k,)  # list of Gaussian likelihoods P(x| s_i=1, phi)

    for i in range(k):
        p_x_s[i] = (multivariate_normal.pdf(
            x, mean=means[i], cov=covariances[i]))

    # infer posterior P(ϕ|x) of structural latent variable
    post_phi_s = [np.sum(p_x_s * p_s_phi[j]) for j in range(k)]
    # normalise the posterior
    post_phi_s = post_phi_s / np.sum(np.dot(p_s_phi, p_x_s))

    # plot_posterior_perception('posterior P(ϕ|x) ', 3, post_phi_s)
    # print(f'posterior over ϕ: {post_phi_s}')

    # infer posterior P(S|ϕ, x) given posterior over ϕ as argmax decision
    
    p_s_x = p_x_s * p_s_phi[np.argmax(post_phi_s)] # how do I not have argmax?
   
    # normalise posterior
    post_s_x = p_s_x / np.sum(p_x_s * p_s_phi[np.argmax(post_phi_s)])
    
    return post_s_x

def minmax_norm(arr):
    arr = np.array(arr, dtype=float)
    mn, mx = np.min(arr), np.max(arr)
    return (arr - mn) / (mx - mn + 1e-12)

def negentropy_01(p):
    """Return negentropy in [0,1] : 1 - (H(p)/log k)."""
    p = np.asarray(p, dtype=float)
    k = p.size
    eps = 1e-12
    H = -np.sum(p * np.log(p + eps))
    #H_norm = H / np.log(k)
    return 1.0 - H

def metrics_from_posterior(p):
    p = np.asarray(p, dtype=float)
    ent = negentropy_01(p)
    p_sorted = np.sort(p)[::-1]
    maxp = p_sorted[0]
    diff = maxp - (p_sorted[1] if p_sorted.size > 1 else 0.0)
    return ent, maxp, diff

k = 3
task_dim = 2
n_samples = 1  
attended_states = 2
config = [-96,0,96]
sigma = 64*64
means = np.array([[config[0], 0], [config[1], 0], [config[2], 0]])
    
# p_s_phi = construct_structural_variable_likelihoods(k, attended_states = 2)
p_s_phi = np.array([[1,1,0],
                    [1,0,1],
                    [0,0,1]])


covariances = np.array([[[sigma, 0], [0, sigma]],        
                        [[sigma, 0], [0, sigma]],      
                        [[sigma, 0], [0, sigma]]])

#x = np.array([[-0.5,0]])  # Observed data point


# --------------------------
# Sweep x along horizontal axis
# --------------------------
epsilon = 0.1
x_min = np.min(means[:,0]) - epsilon
x_max = np.max(means[:,0]) + epsilon
xs = np.linspace(x_min, x_max, 201)  # 201 points for smooth curves
y_const = 0.0

# Storage for metrics
ent_norm_list, max_norm_list, diff_norm_list = [], [], []
ent_attn_list, max_attn_list, diff_attn_list = [], [], []

for x_val in xs:
    x = np.array([[x_val, y_const]])
    
    posterior = posterior_likelihood_norm(x, k, means, covariances, T=1)
    # print(f'posterior no attention: {posterior}')
    # plot_posterior_perception('Posterior with no attention', k, posterior)
    ent_n, max_n, diff_n = metrics_from_posterior(posterior)
    ent_norm_list.append(ent_n)
    max_norm_list.append(max_n)
    diff_norm_list.append(diff_n)

    post_attn = posterior_with_attention_beta(x, k, means, covariances, p_s_phi)
    #post_attn = posterior_with_attention_argmax(x, k, means, covariances, p_s_phi)
    ent_a, max_a, diff_a = metrics_from_posterior(post_attn)
    ent_attn_list.append(ent_a)
    max_attn_list.append(max_a)
    diff_attn_list.append(diff_a)


# Normalize for no-attention posterior
max_norm_list = minmax_norm(max_norm_list)
diff_norm_list = minmax_norm(diff_norm_list)
# ent already lies in [0,1], leave as is
ent_norm_list_n = minmax_norm(ent_norm_list)

# Normalize for attention posterior
max_attn_list = minmax_norm(max_attn_list)
diff_attn_list = minmax_norm(diff_attn_list)
ent_attn_list = minmax_norm(ent_attn_list)

# Plot metrics for posterior_likelihood_norm
plt.figure(figsize=(7,4))
plt.plot(xs, diff_norm_list, label='diff')
plt.plot(xs, max_norm_list, label='max')
plt.plot(xs, ent_norm_list, label='ent')
plt.xlabel('x (horizontal axis of observation)')
plt.ylabel('metric value')
plt.title('Metrics vs x — no attention')
plt.legend()
plt.tight_layout()
plt.show()

# Plot metrics for posterior_with_attention
plt.figure(figsize=(7,4))
plt.plot(xs, diff_attn_list, label='diff')
plt.plot(xs, max_attn_list, label='max')
plt.plot(xs, ent_attn_list, label='ent')
plt.xlabel('x (horizontal axis of observation)')
plt.ylabel('metric value')
plt.title('Metrics vs x — with attention')
plt.legend()
plt.tight_layout()
plt.show()

#print(f'posterior with attention: {post}')

# plot_posterior_perception('Posterior with attention', k, post)
# plt.show()