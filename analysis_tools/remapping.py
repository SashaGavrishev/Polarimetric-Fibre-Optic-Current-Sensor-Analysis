# -*- coding: utf-8 -*-
"""Remapping of ellipse to circle."""

import numpy as np


def map_ellipse(alpha, beta, phi, cent, s_alpha, s_beta, s_phi, s_cent, x, y):
    """
    Use an ellipse's geometric parameters, to remap ellipse to a unit circle.

    Given ellipse data points X = [x_, y_]' where x_ and y_ are 1D arrays,
    can always offset, then rotate, then scale an ellipse to reduce it to
    a unit circle.

    Args:
        alpha (float): alpha scaling factor
        beta (float): beta scaling factor
        phi (float): rotation angle [rad]
        cent (np.ndarray): centre vector of ellipse
        s_alpha (float): 1 standard deviation error of alpha
        s_beta (float): 1 std error of beta
        s_phi (float): 1 std error of phi [rad]
        s_cent (np.ndarray): 1 std error of each centre vector
        x (np.ndarray): ellipse x-data
        y (np.ndarray): ellipse y-data

    Returns:
        result (np.ndarray): remapped ellipse points to a unit circle
        std (np.ndarray): 1 std error of each remapped ellipse points

    """
    X = np.array([x, y])

    cent = np.array([[cent[0]], [cent[1]]])

    R_phi = np.array([[np.cos(phi), - np.sin(phi)],
                     [np.sin(phi), np.cos(phi)]])

    dR_phi_dphi = np.array([[-np.sin(phi), - np.cos(phi)],
                     [np.cos(phi), -np.sin(phi)]])
    scale = np.array([[1/alpha, 0], [0, 1/beta]])

    result = (scale @ R_phi @ (X - cent))

    df_dphi = scale @ dR_phi_dphi @ (X-cent)

    df_dx0 = scale @ R_phi @ (-1 * np.array([[cent[0][0]], [0]]))
    df_dy0 = scale @ R_phi @ (-1 * np.array([[0], [cent[1][0]]]))

    df_dscale = -1 * np.array([[1/alpha, 0], [0, 1/beta]]) @ result

    s_scale = np.array([[s_alpha],[s_beta]])
    error = np.square(df_dphi) * s_phi **2 + np.square(df_dx0) * s_cent[0] **2 + \
            np.square(df_dy0) * s_cent[1] ** 2 + np.square(df_dscale) * s_scale **2
    std = np.sqrt(error)

    return result, std
