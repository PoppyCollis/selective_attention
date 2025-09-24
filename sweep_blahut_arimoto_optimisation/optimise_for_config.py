# Blahut-Arimoto Optimisation

import numpy as np
np.set_printoptions(precision=6, suppress=True)
import matplotlib.pyplot as plt

from utils import *
from blahut_arimoto import *


def main():
    # --- Run BA with history tracking on the same quadratic-utility example ---
    X = 3
    A = 3
    mu_as = np.array([-96, 0, 96])
    sigma_as = np.ones(A)*(64)
    epsilon = 64
    L, Uhi = mu_as[0] - epsilon, mu_as[-1] + epsilon

    beta1 = np.inf
    beta2, beta3 = 1,2
    
    # Grid sampling 
    n_samples = 300
    w, pw = make_w_samples(L, Uhi, n_samples, grid=True)
    U_pre = build_U_pre(U_fn, A, w, mu_as, sigma_as)

    res = threevar_BA_iterations(
        X=X, beta1=beta1, beta2=beta2, beta3=beta3,
        U_pre=U_pre, pw=pw, tol=1e-10, maxiter=50,
        init_pogw_uniformly=False, init_pogw_sparse=True, init_pagow_uniformly=True,
        track_history=True
    )

    perf_df = collect_metrics_over_history(res.history, pw, U_pre, beta1, beta2, beta3)
    
if __name__== "__main__":
    main()