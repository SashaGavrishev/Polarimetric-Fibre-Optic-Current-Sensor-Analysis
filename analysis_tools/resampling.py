# -*- coding: utf-8 -*-
"""Functions to resample data."""

# Import Libraries

import numpy as np
from scipy.interpolate import CubicSpline


def resample_cs(t, x_data, y_data):
    """Resamples x y data using a cubic spline."""
    y = np.c_[x_data, y_data]
    cs = CubicSpline(
        t,
        y,
    )
    return cs


def twod_hist(x_data, y_data, bins=50):
    """Resample 2D data using a histogram."""
    Z, xedges, yedges = np.histogram2d(x_data, y_data, bins=bins)

    x_c = np.convolve(xedges, np.ones(2) / 2, mode="valid")
    y_c = np.convolve(yedges, np.ones(2) / 2, mode="valid")

    x_width = xedges[1] - xedges[0]
    y_width = yedges[1] - xedges[0]

    Pts = np.vstack((x_c, y_c))

    argZ = np.argwhere(Z != 0).T

    x_c, y_c = Pts[[[0], [1]], argZ]

    return x_c, y_c, x_width, y_width
