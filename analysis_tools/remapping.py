# -*- coding: utf-8 -*-
"""Remapping of ellipse to circle."""

import numpy as np


def map_ellipse(alpha, beta, phi, cent, x, y):
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
        x : ellipse x-data
        y (np.ndarray): ellipse y-data

    Returns:
        X (np.ndarray): remapped ellipse points to a unit circle

    """
    X = np.array([x, y])

    cent = np.array([[cent[0]], [cent[1]]])

    R_phi = np.array(
        [[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]
    )

    scale = np.array([[1 / alpha, 0], [0, 1 / beta]])

    return scale @ R_phi @ (X - cent)
