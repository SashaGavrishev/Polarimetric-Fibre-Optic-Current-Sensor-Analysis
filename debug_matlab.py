# -*- coding: utf-8 -*-
"""Comparing Direct Fit and FGEE Fit."""

#########################################################################
# Import Packages
#########################################################################

from plotting_tools import (
    algebraic_plot_ellipse,
    plot_circle,
    lissajous_plot,
    current_plot_FP,
    signals_post_filter,
)
from ellipse_fitting.cov_geo_param import cov_geo_params
from ellipse_fitting.fast_guaranteed_estimate import fgee_estimate
from ellipse_fitting.cov_geo_param import cov_geo_params
from ellipse_fitting.ellipse_tools import (
    ellipse_parameters,
    plot_ellipse,
)
from ellipse_fitting.direct_fit import direct_ellipse_fit
from analysis_tools.theta_extractor import theta_t
from analysis_tools.remapping import map_ellipse
from analysis_tools.filtering import filter_sig
from analysis_tools.current_analysis import (
    return_current_F,
    return_current_P,
)
from analysis_tools.resampling import twod_hist
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import re
from time import time

#########################################################################
# Constants and Parameters
#########################################################################

# Plotting Resolution

mpl.rcParams["figure.dpi"] = 300

# Data File Path

path = "data/matlab_ellipse_half.csv"

#########################################################################
# Define Helper Functions
#########################################################################


def read_file_details(file, n, row):
    """Read in part of scope data file and extract a specific row."""
    with pd.read_csv(
        file, chunksize=n, index_col=0, header=None
    ) as reader:
        chuncks = reader
        df = next(chuncks)
        return df.iloc[row].to_numpy()


def extract_number(string):
    """Extract float from string.

    Args:
        string (string): string in format XX.YYSS

    Returns:
        float (float): XX.YY

    """
    numbers = re.findall(r"\d+", string)
    numbers.insert(1, ".")
    return float("".join(numbers))


def import_data(path):
    data = {}

    CH3_V, CH4_V = np.loadtxt(path, delimiter=",", unpack=True)

    data["CH3"] = CH3_V
    data["CH4"] = CH4_V

    return data


def filter_data(data, l1, l3, l4, lim0, lim1, resample=True):
    time = data["time"]
    CH3_V = data["CH3"]
    CH4_V = data["CH4"]
    CH1_V = data["CH1"]

    ch3filt = filter_sig(time, CH3_V)
    ch3filt.rm_DC_offset()
    ch3filt.apply_dwt(l3)

    ch4filt = filter_sig(time, CH4_V)
    ch4filt.rm_DC_offset()
    ch4filt.apply_dwt(l4)

    ch1filt = filter_sig(time, CH1_V)
    ch1filt.rm_DC_offset()
    ch1filt.apply_dwt(l1)

    _, CH3_V_f = ch3filt.signal()

    _, CH4_V_f = ch4filt.signal()

    _, CH1_V_f = ch1filt.signal()

    if resample is True:
        CH3_V_ellipse, CH4_V_ellipse = twod_hist(
            CH3_V_f, CH4_V_f, 1000
        )

    else:
        ch3filt.truncate(lim0, lim1)
        ch4filt.truncate(lim0, lim1)

        # ch4filt.cubic_resample(5000)
        # ch3filt.cubic_resample(5000)

        _, CH3_V_ellipse = ch3filt.signal()

        _, CH4_V_ellipse = ch4filt.signal()

    data["CH1f"] = CH1_V_f
    data["CH3f"] = CH3_V_f
    data["CH4f"] = CH4_V_f

    data["CH3e"] = CH3_V_ellipse
    data["CH4e"] = CH4_V_ellipse

    return data


#########################################################################
# Main Code
#########################################################################

# Import Data

data = import_data(path)


# Plot Lissajous

fig2, ax2, line2_1 = lissajous_plot(data["CH3"], data["CH4"])

ax2.set_title("")
ax2.legend()

# Fit

tdir = time()
theta_dir = direct_ellipse_fit(data["CH3"], data["CH4"])
elapsed_time_dir = time() - tdir

tfgee = time()
theta_fgee = fgee_estimate(data["CH3"], data["CH4"], theta_dir)
elapsed_time_fgee = time() - tfgee

# Comparison of Fits

fig3, ax3, line3_1 = lissajous_plot(data["CH3"], data["CH4"])

ax3.set_title("")

line3_2 = algebraic_plot_ellipse(theta_dir, ax3)
line3_3 = algebraic_plot_ellipse(theta_fgee, ax3)

line3_1.set_label("data")

line3_2.set_label("direct fit")
line3_2.set_color("royalblue")

line3_3.set_label("fgee fit")
line3_3.set_color("darkred")

ax3.legend()

print("-----------------------------------------------")
print(f"DIR Method Total Time: {elapsed_time_dir} seconds")
print(f"FGEE Method Total Time: {elapsed_time_fgee} seconds")

# %%

# Errors on FGEE Method

geoCov = cov_geo_params(
    np.array([theta_fgee]).T, data["CH3"], data["CH4"]
)

fgee_geo_errors = np.sqrt(np.diag(geoCov))

alpha, beta, phi, cent = ellipse_parameters(theta_fgee)

xC, yC = cent

A, B, xC, yC, phi = alpha, beta, xC, yC, phi

sig_A, sig_B, sig_xC, sig_yC, sig_phi = fgee_geo_errors

print("-----------------------------------------------")
print("FGEE PARAMS:")
print(f"└──A: {A} ± {sig_A}")
print(f"└──B: {B} ± {sig_B}")
print(f"└──xC: {xC} ± {sig_xC}")
print(f"└──yC: {yC} ± {sig_yC}")
print(f"└──phi: {phi} ± {sig_phi}")
print("-----------------------------------------------")

x_p, y_p = plot_ellipse(
    A + sig_A,
    B + sig_B,
    phi + sig_phi,
    [xC + sig_xC, yC + sig_yC],
    1000,
)

ax3.plot(x_p, y_p)
