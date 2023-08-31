# -*- coding: utf-8 -*-
"""Fast Guaranteed Ellipse Estimate"""

# Import Libraries

import numpy as np
from scipy.linalg import lstsq, pinv
from ellipse_fitting.normalise_isotropically import (
    normalise_isotropically,
)
from ellipse_fitting.fast_guaranteed_fit import fgee_fit


def fgee_estimate(x, y, theta_dir, covList=None):
    dPts = np.array([x, y]).T

    nPts = max(dPts.shape)

    if covList == None:
        covList = np.tile(np.eye(2), (1, nPts)).T.reshape(nPts, 2, 2)

    initialEllipseParameters = theta_dir

    normalizedPoints, T = normalise_isotropically(dPts)

    initialEllipseParameters = (
        initialEllipseParameters
        / np.linalg.norm(initialEllipseParameters)
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

    initialEllipseParametersNormalizedSpace = lstsq(
        E,
        (
            P34
            @ pinv(D3)
            @ np.linalg.inv(np.kron(T, T)).conj().T
            @ D3
            @ P34
            @ E
            @ initialEllipseParameters
        ),
    )[0]

    initialEllipseParametersNormalizedSpace = (
        initialEllipseParametersNormalizedSpace
        / np.linalg.norm(initialEllipseParametersNormalizedSpace)
    )

    normalised_CovList = np.empty((nPts, 2, 2))

    for iPts in range(0, nPts):
        covX_i = np.zeros((3, 3))
        covX_i[0:2, 0:2] = covList[iPts]
        covX_i = T * covX_i * T.conj().T

        normalised_CovList[iPts] = covX_i[0:2, 0:2]

    para = initialEllipseParametersNormalizedSpace

    p = para[1] / (2 * para[0])
    q = (para[2] / para[0] - (para[1] / (2 * para[0])) ** 2) ** (
        1 / 2
    )
    r = para[3] / para[0]
    s = para[4] / para[0]
    t = para[5] / para[0]

    latentParameters = np.array([p, q, r, s, t])

    ellipseParametersFinal, iterations = fgee_fit(
        latentParameters,
        normalizedPoints.conj().T,
        normalised_CovList,
    )

    ellipseParametersFinal = ellipseParametersFinal / np.linalg.norm(
        ellipseParametersFinal
    )

    estimatedParameters = np.linalg.lstsq(
        E,
        (
            P34
            @ np.linalg.pinv(D3)
            @ np.kron(T, T).conj().T
            @ D3
            @ P34
            @ E
            @ ellipseParametersFinal
        ),
        rcond=None,
    )[0]

    estimatedParameters = estimatedParameters / np.linalg.norm(
        estimatedParameters
    )
    estimatedParameters = estimatedParameters / np.sign(
        estimatedParameters[-1]
    )

    return estimatedParameters
