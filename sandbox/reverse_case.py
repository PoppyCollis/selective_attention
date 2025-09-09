import numpy as np
import pandas as pd
import math
from dataclasses import dataclass
from typing import List, Dict

def gaussian_kl(q_m, q_v, p_m, p_v):
    return 0.5 * ((q_v / p_v) + ((q_m - p_m) ** 2) / p_v - 1.0 + math.log(p_v / q_v))

def expected_log_lik_gaussian_known_variance(x, m, v, sigma2):
    return -0.5 * math.log(2 * math.pi * sigma2) - 0.5 * ((x - m) ** 2 + v) / sigma2

@dataclass
class ModelSpec:
    sigma: float
    tau: float
    m0: List[float]
    pi: List[float]

@dataclass
class PosteriorComponent:
    m: float
    v: float
    kl: float

@dataclass
class FitResult:
    elbo: float
    accuracy: float
    complexity: float
    log_pz: float
    posteriors: Dict[int, PosteriorComponent]
    assignments: List[int]

def fit_with_hard_assignments(X: List[float], assignments: List[int], spec: ModelSpec) -> FitResult:
    K = len(spec.m0)
    sigma2 = spec.sigma ** 2
    tau2 = spec.tau ** 2

    X_by_k = {k: [] for k in range(K)}
    for x, z in zip(X, assignments):
        X_by_k[z].append(x)

    post = {}
    total_kl = 0.0
    total_acc = 0.0
    for k in range(K):
        n_k = len(X_by_k[k])
        if n_k == 0:
            v_k = tau2
            m_k = spec.m0[k]
        else:
            v_k = 1.0 / (1.0 / tau2 + n_k / sigma2)
            m_k = v_k * (spec.m0[k] / tau2 + sum(X_by_k[k]) / sigma2)
        kl_k = gaussian_kl(m_k, v_k, spec.m0[k], tau2)
        post[k] = PosteriorComponent(m=m_k, v=v_k, kl=kl_k)
        total_kl += kl_k

    for x, z in zip(X, assignments):
        pc = post[z]
        total_acc += expected_log_lik_gaussian_known_variance(x, pc.m, pc.v, sigma2)

    log_pz = sum(math.log(spec.pi[z]) for z in assignments)
    elbo = total_acc - total_kl + log_pz
    return FitResult(elbo, total_acc, total_kl, log_pz, post, assignments)

def run_minimal():
    X = [-2.1, 0.1, 2.0]
    spec_full = ModelSpec(sigma=0.3, tau=0.1, m0=[-2.0, 0.0, 2.0], pi=[1/3, 1/3, 1/3])
    fit_full = fit_with_hard_assignments(X, [0,1,2], spec_full)

    spec_red = ModelSpec(sigma=0.3, tau=0.1, m0=[-2.0, 0.0], pi=[1/2, 1/2])
    fit_red = fit_with_hard_assignments(X, [0,1,1], spec_red)

    def summarize(tag, fit):
        return {
            "model": tag,
            "ELBO": fit.elbo,
            "Accuracy": fit.accuracy,
            "Complexity": fit.complexity,
            "log p(z)": fit.log_pz,
            "assignments": fit.assignments,
            "q(mu0)_m": fit.posteriors[0].m, "q(mu0)_v": fit.posteriors[0].v, "q(mu0)_KL": fit.posteriors[0].kl,
            "q(mu1)_m": fit.posteriors[1].m, "q(mu1)_v": fit.posteriors[1].v, "q(mu1)_KL": fit.posteriors[1].kl,
            **({"q(mu2)_m": fit.posteriors[2].m, "q(mu2)_v": fit.posteriors[2].v, "q(mu2)_KL": fit.posteriors[2].kl} if 2 in fit.posteriors else {})
        }

    df = pd.DataFrame([summarize("Full (3 comps, correct)", fit_full),
                       summarize("Reduced (2 comps, pruned)", fit_red)])
    return df

if __name__ == "__main__":
    df = run_minimal()
    print(df.to_string(index=False))