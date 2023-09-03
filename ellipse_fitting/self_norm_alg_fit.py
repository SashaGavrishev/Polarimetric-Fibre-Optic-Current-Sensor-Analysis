# -*- coding: utf-8 -*-
"""Self-Normalizing Algebraic Method"""

import numpy as np


def self_norm_alg_fit(x, y, debias=True):
    """Self-Normalizing Algebraic Method"""
    x, y = np.array(x), np.array(y)

    D = np.array([x * x, x * y, y * y, x, y, np.ones_like(x)])
    S = np.inner(D, D)

    S12 = S[5, :5]
    S22 = S[5, 5]
    Sr = S[:5, :5] - np.outer(S12, S12) / S22
    C = np.array(
        [
            [4 * S12[0], 2 * S12[1], 0, 2 * S12[3], 0],
            [2 * S12[1], S12[0] + S12[2], 2 * S12[1], S12[4], S12[3]],
            [0, 2 * S12[1], 4 * S12[2], 0, 2 * S12[4]],
            [2 * S12[3], S12[4], 0, S22, 0],
            [0, S12[3], 2 * S12[4], 0, S22],
        ]
    )

    L = np.linalg.inv(np.linalg.cholesky(C))
    evals, evects = np.linalg.eigh(np.inner(np.dot(L, Sr), L))
    j = np.argmin(evals)
    G1 = np.dot(evects[:, j], L)
    G2 = -np.dot(G1, S12)/S22
    if debias:
        G2+= (G1[0] + G1[2]) * evals[j]
    
    return np.r_[G1,G2]
