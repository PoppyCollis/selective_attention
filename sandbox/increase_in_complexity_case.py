import numpy as np


# post_full = np.array([0.6,0.35,0.05])
# prior_full = np.array([1/3,1/3,1/3])

post_full = np.array([0.28,0.28,0.44])
prior_full = np.array([1/3,1/3,1/3])

ln_q_p_full = np.log(post_full/prior_full)
complexity_full = np.dot(post_full, ln_q_p_full)
print(complexity_full)

post_red = np.array([0.28/(0.28+0.44),0.44/(0.28+0.44)])
prior_red = np.array([0.5,0.5])

ln_q_p = np.log(post_red/prior_red)
complexity_red = np.dot(post_red, ln_q_p)
print(complexity_red)


"""
So KL doesn't always drop: when the discarded state still carries 
non-trivial mass and/or the remaining two become imbalanced after renormalization, 
the renormalization penalty can outweigh the fixed log(3/2) bonus from comparing to a smaller prior.

Short answer: if you prune the least-likely state, the KL to a uniform prior can go up, but by a tiny bounded amount
The KL can increase, but by < 0.001 nats in the worst case under “drop the least likely”.

The accuracy gain from conditioning on the top two can be arbitrarily large when the dropped option is much worse.
"""