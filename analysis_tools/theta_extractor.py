# -*- coding: utf-8 -*-
"""Code to extra theta(t) from unit circle data."""

import numpy as np


def theta_t(time, x_remap, y_remap):
    """Extract theta(t) from remapped ellipse data."""
    theta_0 = np.arctan2(x_remap[0], y_remap[0])
    theta = [0]
    last = 0

    for i, t in enumerate(time):
        if i != 0:
            theta_t = np.arctan2(x_remap[i], y_remap[i])
            theta_t = (
                (theta_t - last + np.pi) % (2 * np.pi) - np.pi + last
            )
            last = theta_t
            theta.append(0.5 * theta_t - 0.5 * theta_0)

    return time, theta
