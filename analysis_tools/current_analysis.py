# -*- coding: utf-8 -*-
"""Current Sensor Analysis Code."""

# Import Libraries
import numpy as np


def return_current_F(
    N_fibre, N_coil, S, V, sign, theta, sig_N_fibre, sig_N_coil, sig_V, sig_theta
):
    """Convert angle data to current data, Faraday mirror version."""
    theta = np.asarray(theta)
    sig_theta = np.asarray(sig_theta)
    current = theta / (V * S * N_coil * N_fibre * sign * 2)
    sig_current = np.sqrt(
        (((theta) / (V * S * (N_fibre**2) * sign * 2 * N_coil)) ** 2 * sig_N_fibre**2)
        + (((theta) / (N_fibre * S * (V**2) * sign * 2 * N_coil)) ** 2 * sig_V**2)
        + (((1) / (V * S * N_fibre * sign * 2 * N_coil)) ** 2 * sig_theta**2)
        + (((theta) / (V * S * N_fibre * sign * 2 * (N_coil**2))) ** 2 * sig_N_coil**2)
    )
    return current, sig_current


def return_current_P(V_out):
    """Convert oscilloscope signal from Pearson to Current."""
    return V_out * 0.1 * 2  # Pearson is 0.1V per Amp
