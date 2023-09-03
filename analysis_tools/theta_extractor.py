# -*- coding: utf-8 -*-
"""Code to extra theta(t) from unit circle data."""

import numpy as np


def theta_t(time, x_remap, y_remap, x_std, y_std):
    """
    Extract theta(t) from remapped ellipse data.

    Args:
        time (np.ndarray): time data
        x_remap (np.ndarray): remapped x-data
        y_remap (np.ndarray): remapped y-data
        x_std (np.ndarray): 1 standard deviation error of each
            remapped x-data point
        y_std (np.ndarray): 1 std error of each remapped x-data point

    Returns:
        time (np.ndarray): time data, unchanged
        theta (np.ndarray): theta(t) data
        theta_std (np.ndarray): 1 std error of each theta(t) data
    """
    theta_0 = np.arctan2(x_remap[0], y_remap[0])
    theta = [0]
    last = 0

    for i, t in enumerate(time):
        if i != 0:
            try:
                theta_t = np.arctan2(x_remap[i], y_remap[i])

                theta_t = (theta_t - last + np.pi) % (2 * np.pi) - np.pi + last

                last = theta_t

                theta.append(0.5 * theta_t - 0.5 * theta_0)
            except:
                print(x_remap, y_remap)
                break

    df_dx = 1 / (1 + (x_remap / y_remap) ** 2) * 1 / y_remap
    df_dy = 1 / (1 + (x_remap / y_remap) ** 2) * (-x_remap / y_remap**2)

    theta_std_0 = np.sqrt(
        np.square(df_dx[0]) * np.square(x_std[0])
        + np.square(df_dy[0]) * np.square(y_std[0])
    )

    theta_std = np.sqrt(
        np.square(df_dx) * np.square(x_std) + np.square(df_dy) * np.square(y_std)
    )
    theta_std = np.sqrt((0.5 * theta_std) ** 2 + (0.5 * theta_std_0) ** 2)

    return time, theta, theta_std
