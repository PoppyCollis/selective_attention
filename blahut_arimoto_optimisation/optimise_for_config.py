# Blahut-Arimoto Optimisation

import numpy as np
np.set_printoptions(precision=6, suppress=True)
import matplotlib.pyplot as plt

from utils import *
from blahut_arimoto import *

def main():
    # --- Run BA with history tracking on the same quadratic-utility example ---
    X = 2
    A = 3
    mu_as = np.array([-96, 0, 96])
    sigma_as = np.ones(A)*(64)
    epsilon = 64
    L, Uhi = mu_as[0] - epsilon, mu_as[-1] + epsilon

    beta1, beta2, beta3 = np.inf,0.72,0.63

    # Grid sampling for determinism
    n_samples = 300
    # w, pw = make_w_samples_gaussian(L, Uhi, n_samples, grid=True)
    w, pw = make_w_samples(L, Uhi, n_samples, grid=True)

    U_pre = build_U_pre(U_fn, A, w, mu_as, sigma_as)


    res = threevar_BA_iterations(
        X=X, beta1=beta1, beta2=beta2, beta3=beta3,
        U_pre=U_pre, pw=pw, tol=1e-10, maxiter=100,
        init_pogw_uniformly=False, init_pogw_sparse=True, init_pagow_uniformly=True,
        track_history=True
    )

    perf_df = collect_metrics_over_history(res.history, pw, U_pre, beta1, beta2, beta3)

    # Show the table and plots
    print(perf_df.iloc[-1])

    fig1, fig2 = plot_convergence(perf_df)

    #print("p(x|w):\n", res.pogw)
    print("p(x):", res.po)
    #print("p(a|x,w):\n", res.pagow)
    print("p(a|x):\n", res.pago)
    print("p(a):", res.pa)
    #print("p(a|w):", res.pagw)

    figs = show_prob_matrix(res.pago, name="p(a|x)")
    plt.show()

    # Display a 3D matrix with slices
    for fig in show_prob_matrix(res.pogw, name="p(x|,w)"):
        plt.show()
        
    figs = show_prob_matrix(res.pagw, name="p*(a|w)")
    plt.show()

    pagw_stats = compute_pagw_stats_over_w(res.pagw)

    pagw_df    = pagw_stats_df(pagw_stats, w)     
    fig_all    = plot_pagow_stats_vs_w_combined(w, pagw_stats)
    plt.show()

    df = pagw_stats_df(pagw_stats, w)
    
if __name__== "__main__":
    main()