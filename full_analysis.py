# -*- coding: utf-8 -*-
"""An Example Analysis of FOCS Data."""

########################################################################################
# Import Packages
########################################################################################

from helper_function.plotting_tools import (
    algebraic_plot_ellipse,
    lissajous_plot,
    lissajous_plot_errors,
    signals_post_filter,
    plot_circle,
    current_plot_FP,
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
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import re
from time import time

########################################################################################
# Constants and Parameters
########################################################################################

# Plotting Resolution

mpl.rcParams["figure.dpi"] = 300

# Data File Path

path = "data/large_proportion_of_ellipse.csv"

########################################################################################
# Define Helper Functions
########################################################################################


def read_file_details(file, n, row):
    """Read in part of scope data file and extract a specific row."""
    with pd.read_csv(file, chunksize=n, index_col=0, header=None) as reader:
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

    scope_time_interval = extract_number(read_file_details(path, 10, 8)[1])  # uS

    index, CH1_V, CH3_V, CH4_V = np.loadtxt(
        path, delimiter=",", skiprows=11, unpack=True
    )

    time = index * scope_time_interval

    data["index"] = index
    data["time"] = time
    data["CH1"] = CH1_V
    data["CH3"] = CH3_V
    data["CH4"] = CH4_V

    return data


def filter_data(data, l1, l3, l4, resample=1000):
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

    CH3_V_ellipse, CH4_V_ellipse = twod_hist(CH3_V_f, CH4_V_f, resample)

    data["CH1f"] = CH1_V_f
    data["CH3f"] = CH3_V_f
    data["CH4f"] = CH4_V_f

    data["CH3e"] = CH3_V_ellipse
    data["CH4e"] = CH4_V_ellipse

    return data


def multipage(filename, figs=None, dpi=100):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format="pdf")
    pp.close()


#########################################################################
# Main Code
#########################################################################

# Import Data

data = import_data(path)

data = filter_data(data, 8, 8, 8, 1000)

# Raw and Filtered Signals

fig1, ax1 = signals_post_filter(
    data["time"], data["CH3"], data["CH4"], data["CH3f"], data["CH4f"]
)

# Plot Lissajous of Filtered Data

fig2, ax2, line2_1 = lissajous_plot(data["CH3f"], data["CH4f"])

line2_1.set_label("filtered data")

ax2.set_title("")
ax2.legend()

# Fit

tdir = time()
theta_dir = direct_ellipse_fit(data["CH3e"], data["CH4e"])
elapsed_time_dir = time() - tdir

tfgee = time()
theta_fgee = fgee_estimate(data["CH3e"], data["CH4e"], theta_dir)
elapsed_time_fgee = time() - tfgee

# Comparison of Fits

fig3, ax3, line3_1 = lissajous_plot(data["CH3e"], data["CH4e"])

ax3.set_title("")

line3_2 = algebraic_plot_ellipse(theta_dir, ax3)
line3_3 = algebraic_plot_ellipse(theta_fgee, ax3)

line3_1.set_marker("x")
line3_1.set_linestyle("")
line3_1.set_label("filtered data")

line3_2.set_label("direct fit")
line3_2.set_color("royalblue")

line3_3.set_label("fgee fit")
line3_3.set_color("darkred")

ax3.legend()

print("-----------------------------------------------")
print(f"DIR Method Total Time: {elapsed_time_dir} seconds")
print(f"FGEE Method Total Time: {elapsed_time_fgee} seconds")

# Errors on FGEE Method

geoCov = cov_geo_params(np.array([theta_fgee]).T, data["CH3e"], data["CH4e"])

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

# Remapping
sig_cent = np.array([sig_xC, sig_yC])

dPts, std_dPts = map_ellipse(
    A, B, phi, cent, sig_A, sig_B, sig_phi, sig_cent, data["CH3f"], data["CH4f"]
)

x_remap, y_remap = dPts
sig_x_remap, sig_y_remap = std_dPts

fig4, ax4, line4_1 = lissajous_plot(x_remap, y_remap)

ax4.set_title("Lissajous Figure - Remapped")
plot_circle(ax4)
ax4.set_xlim(-2, 2)
ax4.set_ylim(-2, 2)
ax4.legend()

theta_time, theta, sig_theta = theta_t(
    data["time"], x_remap, y_remap, sig_x_remap, sig_y_remap
)

N_fibre = 5
N_coil = 81
S = +1
V = 0.71e-6
sign = -1
sig_N_fibre = 0.5
sig_N_coil = 0.5
sig_V = 0.03e-6


current_f, sig_current_f = return_current_F(
    N_fibre, N_coil, S, V, sign, theta, sig_N_fibre, sig_N_coil, sig_V, sig_theta
)

current_p = return_current_P(data["CH1"])

fig5, ax5 = current_plot_FP(
    data["time"], current_f, sig_current_f, data["time"], current_p
)

multipage("output.pdf")

plt.show()
