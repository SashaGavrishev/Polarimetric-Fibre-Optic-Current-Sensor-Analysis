# -*- coding: utf-8 -*-
"""Current Sensor Analysis Code."""

# Import Libraries
import numpy as np


def return_current_F(N_coil, N_fibre, S, V, sign, theta):
    """Convert angle data to current data, Faraday mirror version."""
    current = np.asarray(theta) / (
        V * S * N_coil * N_fibre * sign * 2
    )
    return current


def return_current_P(V_out):
    """Convert oscilloscope signal from Pearson to Current."""
    return V_out * 0.1 * 2  # Pearson is 0.1V per Amp
