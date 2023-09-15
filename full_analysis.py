# -*- coding: utf-8 -*-
"""An Example Analysis of FOCS Data."""

########################################################################################
# Import Packages
########################################################################################


from helper_function.plotting_tools import (
    algebraic_plot_ellipse,
    lissajous_plot,
    signals_post_filter,
    plot_circle,
    current_plot_FP,
)
from ellipse_fitting.cov_geo_param import cov_geo_params
from ellipse_fitting.fast_guaranteed_estimate import fgee_estimate
from ellipse_fitting.cov_geo_param import cov_geo_params
from ellipse_fitting.ellipse_tools import (
    ellipse_parameters,
)
from ellipse_fitting.normalise_isotropically import normalise_isotropically
from ellipse_fitting.direct_fit import direct_ellipse_fit
from ellipse_fitting.self_norm_alg_fit import self_norm_alg_fit
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

path = r"data\large_proportion_of_ellipse.csv"

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


def filter_data(data, l1, l3, l4, resample=100):
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

    CH3_V_ellipse, CH4_V_ellipse, _, _ = twod_hist(CH3_V_f, CH4_V_f, resample)
    # CH3_V_ellipse, CH4_V_ellipse = CH3_V_f, CH4_V_f

    data["CH1f"] = CH1_V_f
    data["CH3f"] = CH3_V_f
    data["CH4f"] = CH4_V_f

    data["CH3e"] = CH3_V_ellipse
    data["CH4e"] = CH4_V_ellipse

    return data


def calc_residual_systematic_err(x_remap, y_remap):
    vector = np.array([x_remap, y_remap]).T
    length = np.linalg.norm(vector, axis=1)
    residual_error = np.abs(length - 1)
    sig_theta_systematic = np.arctan2(
        np.sqrt(1 - (1 - (residual_error**2) / 2) ** 2),
        (1 - (residual_error**2) / 2),
    )
    return sig_theta_systematic


def multipage(filename, figs=None, dpi=100):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format="pdf")
    pp.close()


#####################################################################################
# Main Code
#####################################################################################

# Import Data

data = import_data(path)

data = filter_data(data, 8, 8, 8, 5000)

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
ax3.set_xlabel(r"$V_{PD1}(t)$ [mV]")
ax3.set_ylabel(r"$V_{PD2}(t)$ [mV]")

line3_2 = algebraic_plot_ellipse(theta_dir, ax3)
line3_4 = algebraic_plot_ellipse(theta_fgee, ax3)

line3_1.set_marker("x")
line3_1.set_linestyle("")
line3_1.set_label("filtered data")

line3_2.set_label("DIRECT fit")
line3_2.set_color("royalblue")


line3_4.set_label("FGEE fit")
line3_4.set_color("darkred")

ax3.legend()

plt.savefig("fitcomparison.png", dpi=300)

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

ax4.set_title("")
plot_circle(ax4)
ax4.set_xlim(-2, 2)
ax4.set_ylim(-2, 2)
ax4.legend()
ax4.set_xlabel(r"Remap$(V_{PD1}(t))$ [mV]")
ax4.set_ylabel(r"Remap$(V_{PD2}(t))$ [mV]")
plt.savefig("remapped.png", dpi=300)

theta_time, theta, sig_theta = theta_t(
    data["time"], x_remap, y_remap, sig_x_remap, sig_y_remap
)

sig_theta_sys = calc_residual_systematic_err(x_remap, y_remap)

sig_theta = np.sqrt((sig_theta) ** 2 + (sig_theta_sys) ** 2)

N_fibre = 10
N_coil = 100
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


def find_signal_start(t, x, thresh=50):
    """Find time of start of signal."""
    i_max = np.argmax(abs(x) > thresh)
    return t[i_max]


def find_signal_end(t, x, thresh=50):
    """Find time of start of signal."""
    i_max = np.argmax(abs(x[::-1]) > thresh)
    return t[-i_max]


t0 = find_signal_start(data["time"], current_f, 50)
t1 = find_signal_end(data["time"], current_f, 50)

# fig1.tight_layout()
ax1[0].margins(0, 0.1)
ax1[1].margins(0, 0.1)
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
ax5.set_xlim(t0 - 100, t1 + 100)
ax5.margins(0.5, 0.1)
fig5.tight_layout()

plt.savefig("current.png", dpi=300)

multipage("output.pdf")

plt.show()
