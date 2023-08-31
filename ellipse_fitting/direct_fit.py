# -*- coding: utf-8 -*-
"""
Fit and extract anlaytically ellipses from 2D data.

Reference:
    https://betterprogramming.pub/least-squares-ellipse-fitting-4c039d7645
"""

import numpy as np


def direct_ellipse_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return fitted ellipse coefficients.

    Based on an algorithm described in 'Numerically Stable Direct Least Squares
    Fitting of Ellipses' (Halir and Flusser, 1998).

    Representing a conic in the form of ax^2 + bxy +cy^2 + dx + ey + f = 0 and
    applying the constraint that b^2 - 4ac < 0 for an ellipse, the code
    calculates the coefficients theta = (a,b,c,d,e,f) needed to fit the ellipse
    using a least squares method.

    Args:
        x (np.ndarray): x-data
        y (np.ndarray): y-data

    Returns:
        theta (np.ndarray): ellipse equation coefficients

    """
    x, y = np.array(x), np.array(y)

    D1 = np.vstack([x**2, x * y, y**2]).T  # first design matrix
    D2 = np.vstack([x, y, np.ones(len(x))]).T  # second design matrix

    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2

    C1 = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]], dtype=float)

    M = np.linalg.inv(C1) @ (
        S1 - S2 @ np.linalg.inv(S3) @ S2.T
    )  # reduced scatter matrix

    eigval, eigvec = np.linalg.eig(M)

    constraint = (4 * eigvec[0] * eigvec[2]) - (eigvec[1] * eigvec[1])

    # eigenvector corresponding to minimal non-negative eigenvalue
    theta1 = eigvec[:, np.nonzero(constraint > 0)[0]]

    theta = np.array(
        [theta1, (-np.linalg.inv(S3) @ np.transpose(S2)) @ theta1]
    )

    return theta.ravel()
