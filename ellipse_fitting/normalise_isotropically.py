# -*- coding: utf-8 -*-
"""Normalising Data Isotropically"""


import numpy as np


def normalise_isotropically(dPts):
    nPts, _ = np.shape(dPts)

    points = np.append(dPts, np.ones((nPts, 1)), axis=1).conj().T

    meanX = points[0, :].mean()
    meanY = points[1, :].mean()

    s = np.sqrt(
        (
            (1 / (2 * nPts))
            * np.sum(
                (points[0, :] - meanX) ** 2
                + (points[1, :] - meanY) ** 2  # changed from 2 to 1
            )
        )
    )

    T = np.array(
        [
            [s**-1, 0, -(s**-1) * meanX],
            [0, s**-1, -(s**-1) * meanY],
            [0, 0, 1],
        ]
    )

    normalizedPts = T @ points

    normalizedPts = normalizedPts.conj().T

    normalizedPts = np.delete(normalizedPts, -1, axis=1)

    return normalizedPts, T
