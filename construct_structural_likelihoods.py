import numpy as np
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



k = 3
p_s_phis = generate_p_s_phis(k)

for n, mat in p_s_phis.items():
    print(f"p_s_phi_{n} =\n{mat}\n")


p_s_phi_1 = p_s_phis[1]
p_s_phi_2 = p_s_phis[2]
p_s_phi_3 = p_s_phis[3]


