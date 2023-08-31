# -*- coding: utf-8 -*-
"""Compute Covariance of Sampson Estimate"""

# Import Libraries

import numpy as np
from scipy.linalg import lstsq, pinv
from ellipse_fitting.normalise_isotropically import (
    normalise_isotropically,
)


def estimate_noise_lvl(algebraicEllipseParameters, dPts, covList):
    t = algebraicEllipseParameters
    t = t / np.linalg.norm(t)
    nPts = max(dPts.shape)
    M = np.zeros((6, 6))
    aml = 0

    for i in range(0, nPts):
        m = dPts[i, :]

        ux_i = (
            np.array(
                [[m[0] ** 2, m[0] * m[1], m[1] ** 2, m[0], m[1], 1]]
            )
            .conj()
            .T
        )

        dux_i = (
            np.array(
                [
                    [2 * m[0], m[1], 0, 1, 0, 0],
                    [0, m[0], 2 * m[1], 0, 1, 0],
                ]
            )
            .conj()
            .T
        )

        A = np.outer(ux_i, ux_i)

        covX_i = covList[i]

        B = dux_i @ covX_i @ dux_i.conj().T

        tBt = t.conj().T @ B @ t
        tAt = t.conj().T @ A @ t

        aml = aml + np.abs(tAt[0][0] / tBt[0][0])
        M = M + A / (tBt)

    sigma_squared = aml / (nPts - 5)

    return sigma_squared


def cov_sampson_estimate(
    algebraicEllipseParameters, dPts, covList=None
):
    nPts = max(dPts.shape)

    if covList is None:
        covList = np.tile(np.eye(2), (1, nPts)).T.reshape(nPts, 2, 2)
        sigma_squared = estimate_noise_lvl(
            algebraicEllipseParameters, dPts, covList
        )

        for i in range(0, nPts):
            covList[i] = covList[i] * sigma_squared

    dPts, T = normalise_isotropically(dPts)

    algebraicEllipseParameters = (
        algebraicEllipseParameters
        / np.linalg.norm(algebraicEllipseParameters)
    )

    E = np.diag([1, 2**-1, 1, 2**-1, 2**-1, 1])

    P34 = np.kron(np.diag([0, 1, 0]), [[0, 1], [1, 0]]) + np.kron(
        np.diag([1, 0, 1]), [[1, 0], [0, 1]]
    )

    D3 = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    algebraicEllipseParametersNormalizedSpace = lstsq(
        E,
        (
            P34
            @ pinv(D3)
            @ np.linalg.inv(np.kron(T, T)).conj().T
            @ D3
            @ P34
            @ E
            @ algebraicEllipseParameters
        ),
    )[0]

    algebraicEllipseParametersNormalizedSpace = (
        algebraicEllipseParametersNormalizedSpace
        / np.linalg.norm(algebraicEllipseParametersNormalizedSpace)
    )

    normalised_CovList = np.empty((nPts, 2, 2))

    for iPts in range(0, nPts):
        covX_i = np.zeros((3, 3))
        covX_i[0:2, 0:2] = covList[iPts]
        covX_i = T * covX_i * T.conj().T

        normalised_CovList[iPts] = covX_i[0:2, 0:2]

    t = algebraicEllipseParametersNormalizedSpace
    t = t / np.linalg.norm(t)
    nPts = max(dPts.shape)
    M = np.zeros((6, 6))

    for i in range(0, nPts):
        m = dPts[i, :]

        ux_i = (
            np.array(
                [[m[0] ** 2, m[0] * m[1], m[1] ** 2, m[0], m[1], 1]]
            )
            .conj()
            .T
        )

        dux_i = (
            np.array(
                [
                    [2 * m[0], m[1], 0, 1, 0, 0],
                    [0, m[0], 2 * m[1], 0, 1, 0],
                ]
            )
            .conj()
            .T
        )

        A = np.outer(ux_i, ux_i)

        covX_i = normalised_CovList[i]

        B = dux_i @ covX_i @ dux_i.conj().T

        tBt = t.conj().T @ B @ t
        tAt = t.conj().T @ A @ t

        M = M + (A / (tBt[0][0]))

    Pt = np.eye(6) - ((t @ t.conj().T) / (np.linalg.norm(t, 2) ** 2))

    U, Ddiag, Vh = np.linalg.svd(M)
    V = Vh.conj().T

    Ddiag = 1 / Ddiag
    Ddiag[5] = 0

    Mm, Mn = np.shape(M)

    D = np.zeros((Mm, Mn))

    np.fill_diagonal(D, Ddiag)

    pinvM = V @ D @ U.conj().T

    covarianceMatrixNormalisedSpace = Pt @ pinvM @ Pt

    F = np.linalg.lstsq(
        E,
        (
            P34
            @ np.linalg.pinv(D3)
            @ np.kron(T, T).conj().T
            @ D3
            @ P34
            @ E
        ),
        rcond=None,
    )[0]

    t = F @ algebraicEllipseParametersNormalizedSpace

    P = np.eye(6) - ((t @ t.conj().T) / (np.linalg.norm(t, 2) ** 2))

    covarianceMatrix = (
        np.linalg.norm(t, 2) ** -2
        * P
        @ F
        @ covarianceMatrixNormalisedSpace
        @ F.conj().T
        @ P
    )

    return covarianceMatrix, covarianceMatrixNormalisedSpace
