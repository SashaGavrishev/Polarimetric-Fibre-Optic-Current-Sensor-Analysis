# -*- coding: utf-8 -*-
"""Functions to Help with Signal Processing."""


import numpy as np
import pywt
from scipy.interpolate import CubicSpline


class filter_sig:
    """Simple class that performs data filtering on a 1D signal."""

    def __init__(self, t, x):
        """Initialise signal and copy for redundancy."""
        self._t, self._t0 = np.array(t), np.array(t)
        self._x, self._x0 = np.array(x), np.array(x)

    def apply_mov_avg(self, width=1):
        """Apply simple moving average filter."""
        if type(width) is not int:
            raise ValueError("width must be an integer")
        else:
            self._x = np.convolve(
                self._x, np.ones(width) / width, mode="same"
            )

    def rm_DC_offset(self):
        """Remove DC offset from data."""
        self._x -= self._x.mean()

    def apply_dwt(self, levels):
        """De-noise via a wavelet transform method."""
        dwt = pywt.wavedec([self._t, self._x], "sym4")
        for i in range(levels - 1):
            dwt[-(i + 1)] = np.zeros_like(dwt[-(i + 1)])
        self._t, self._x = pywt.waverec(dwt, "sym4", mode="symmetric")

    def cubic_resample(self, sample=1000):
        """Resamples x y data using a cubic spline."""
        cs = CubicSpline(self._t, self._x)
        ts = np.linspace(self._t[0], self._t[-1], sample)
        self._t, self._x = ts, cs(ts)

    def cubic_resample_cs(self):
        """Resamples x y data using a cubic spline."""
        return CubicSpline(self._t, self._x)

    def truncate(self, start, stop):
        """Truncate signal range."""
        t_mask_u = np.ma.masked_where(float(start) < self._t, self._t)
        t_mask_l = np.ma.masked_where(float(stop) > self._t, self._t)
        t_mask = (t_mask_u.mask == 1) & (t_mask_l.mask == 1)
        self._t, self._x = self._t[t_mask], self._x[t_mask]

    def signal_reset(self):
        """Restore original signal from copy made in initialisation."""
        self._t = self._t0
        self._x = self._x0

    def signal(self):
        """Output the signal."""
        return self._t, self._x
