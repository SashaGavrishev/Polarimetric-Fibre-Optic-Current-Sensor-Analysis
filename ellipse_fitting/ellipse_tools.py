# -*- coding: utf-8 -*-
"""Ellipse Tools"""

import numpy as np


def ellipse_parameters(theta):
    """
    Convers from conic equation coefficients to ellipse geometric parameters.

    An ellipse can be expressed in the form [R(phi) @ (X - c)]' @
    S(1/alpha^2, 1/beta^2) @ [R(phi) @ (X - c)] = 1, where ' indicates
    transposition and c is the centre of the ellipse, R is a rotation
    matrix, S is a scaling matrix. X is a vector [x, y]'. The function finds
    the necessary parameters for this equation by comparison to the original
    conic equation in matrix form.

    Args:
        theta (np.ndarray): optimised conic coefficients for the ellipse

    Returns:
        alpha (float): alpha scaling factor
        beta (float): beta scaling factor
        phi (float): rotation angle [rad]
        cent (np.ndarray): centre vector of ellipse

    """
    a, b, c, d, e, f = theta  # unpack coefficients

    # from conic equation in matrix form

    Q = np.array([[a, b / 2], [b / 2, c]])

    P = np.array([d, e])

    P = P.T

    # ellipse centre

    cent = -1 / 2 * np.linalg.inv(Q) @ P

    # arbitrary scaling s to allow identification of parameters

    s = cent.T @ Q @ cent - f

    M = 1 / s * Q

    # M encodes the rotation and scaling of the ellipse, it is a symmetric
    # matrix and hence is diagonalisable. The eigenvalues of M give alpha and
    # beta, the eigenvectors are [sin(phi), +/- cos(phi)]' hence give phi.

    eigval, eigvec = np.linalg.eig(M)

    # eigenvectors not necessarily in order
    phi = -np.arctan2(eigvec[0][0], eigvec[0][1])

    alpha = 1 / np.sqrt(
        eigval[1]
    )  # eigenvalues not necessarily in order

    beta = 1 / np.sqrt(eigval[0])

    return alpha, beta, phi, cent


def plot_ellipse(alpha, beta, phi, cent, samples):
    """
    Use the geometric parameters to generate an ellipse.

    Starting with a circle, perform inverse transformations to map a unit
    circle to an ellipse. Unit circle -> Scale -> Rotate -> Offset

    Args:
        alpha (float): alpha scaling factor
        beta (float): beta scaling factor
        phi (float): rotation angle [rad]
        cent (np.ndarray): centre vector of ellipse
        samples (int): number of points for ellipse

    Returns:
        X (np.ndarray): ellipse points

    """
    cent = np.array([[cent[0]], [cent[1]]])

    R_phi = np.array(
        [[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]
    )

    scale = np.array([[1 / alpha, 0], [0, 1 / beta]])

    theta = np.linspace(0, 2 * np.pi, samples)

    X = np.array([np.cos(theta), np.sin(theta)])

    return (np.linalg.inv(R_phi) @ np.linalg.inv(scale) @ X) + cent
