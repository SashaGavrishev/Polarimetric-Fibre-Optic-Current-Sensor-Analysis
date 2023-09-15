# -*- coding: utf-8 -*-
"""Plotting Tools."""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import imageio
from ellipse_fitting.ellipse_tools import (
    plot_ellipse,
    ellipse_parameters,
)
from analysis_tools.resampling import resample_cs


def algebraic_plot_ellipse(theta, ax):
    alpha, beta, phi, cent = ellipse_parameters(theta)
    x_fit, y_fit = plot_ellipse(alpha, beta, phi, cent, 1000)
    return ax.plot(x_fit, y_fit)[0]


def plot_circle(ax):
    """Plot unit circle.

    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): matplotlib axes object

    Returns:
        line2D (matplotlib.lines.Line2D): matplotlib Line2D object

    """
    theta = np.linspace(0, 2 * np.pi, 1000)
    x_c, y_c = np.cos(theta), np.sin(theta)
    return ax.plot(x_c, y_c, "--", label="unit circle")[0]


def lissajous_plot(x, y):
    """Generate a Lissajous Plot.

    Args:
        x (np.ndarray): x data
        y (np.ndarray): y data

    Returns:
        fig (matplotlib.figure.Figure): matplotlib figure object
        ax (matplotlib.axes._subplots.AxesSubplot): matplotlib axes object

    """
    fig, ax = plt.subplots()
    line = ax.plot(x, y, color="gray")[0]
    ax.set_box_aspect(1)
    ax.set_title("Lissajous Figure")
    ax.set_xlabel(r"Signal 1 [A.U.]")
    ax.set_ylabel(r"Signal 2 [A.U.]")
    fig.tight_layout()
    return fig, ax, line


def lissajous_plot_errors(x, y, sig_x, sig_y):
    """Generate a Lissajous Plot with errors.

    Args:
        x (np.ndarray): x data
        y (np.ndarray): y data

    Returns:
        fig (matplotlib.figure.Figure): matplotlib figure object
        ax (matplotlib.axes._subplots.AxesSubplot): matplotlib axes object

    """
    fig, ax = plt.subplots()
    line = ax.errorbar(x, y, xerr=sig_x, yerr=sig_y, fmt=".", capsize=2, color="gray")[
        0
    ]
    ax.set_box_aspect(1)
    ax.set_title("Lissajous Figure")
    ax.set_xlabel(r"Signal 1 [A.U.]")
    ax.set_ylabel(r"Signal 2 [A.U.]")
    return fig, ax, line


def lissajous_plot_T(time, x, y):
    """Generate a Lissajous Plot with time.

    Args:
        x (np.ndarray): x data
        y (np.ndarray): y data

    Returns:
        fig (matplotlib.figure.Figure): matplotlib figure object
        ax (matplotlib.axes._subplots.AxesSubplot): matplotlib axes object

    """
    fig, ax = plt.subplots()
    map1 = ax.scatter(x, y, c=time, cmap="YlGn")
    ax.set_box_aspect(1)
    ax.set_title("Lissajous Figure")
    ax.set_xlabel(r"Signal 1 [A.U.]")
    ax.set_ylabel(r"Signal 2 [A.U.]")
    cbar = fig.colorbar(map1, ax=ax)
    cbar.set_label(r"Time [$\mu{}s$]")
    fig.tight_layout()
    return fig, ax


def current_plot(t_f, current_f, t_p, current_p, t_s, current_s):
    """Generate current plot.

    Args:
        t_f (np.ndarray): time data for Faraday
        current_f (np.ndarray): current data for Faraday
        t_p (np.ndarray): time data for Pearson
        current_p (np.ndarray): current data for Pearson
        t_s (np.ndarray): time data for CASTLE simulation
        current_s (np.ndarray): current data for CASTLE simulation

    Returns:
        fig (matplotlib.figure.Figure): matplotlib figure object
        ax (matplotlib.axes._subplots.AxesSubplot): matplotlib axes object

    """
    fig, ax = plt.subplots()
    ax.plot(t_f, current_f, label="Faraday")
    ax.plot(t_p, current_p, label="Pearson")
    ax.plot(t_s, current_s, label="CASTLE")
    ax.set_xlabel(r"Time [$\mu$s]")
    ax.set_ylabel(r"$I (t)$")
    ax.legend()
    return fig, ax


def current_plot_FP(t_f, current_f, t_p, current_p):
    """Generate current plot.

    Args:
        t_f (np.ndarray): time data for Faraday
        current_f (np.ndarray): current data for Faraday
        t_p (np.ndarray): time data for Pearson
        current_p (np.ndarray): current data for Pearson

    Returns:
        fig (matplotlib.figure.Figure): matplotlib figure object
        ax (matplotlib.axes._subplots.AxesSubplot): matplotlib axes object

    """
    fig, ax = plt.subplots()
    ax.plot(t_p, current_p, "--", label="Pearson", color="black")
    ax.plot(t_f, current_f, label="FOCS", color="royalblue")
    ax.set_xlabel(r"Time [$\mu$s]")
    ax.set_ylabel(r"$I (t)$ [A]")
    ax.legend()
    return fig, ax


def current_plot_FP(t_f, current_f, sig_current_f, t_p, current_p):
    """Generate current plot.

    Args:
        t_f (np.ndarray): time data for Faraday
        current_f (np.ndarray): current data for Faraday
        t_p (np.ndarray): time data for Pearson
        current_p (np.ndarray): current data for Pearson

    Returns:
        fig (matplotlib.figure.Figure): matplotlib figure object
        ax (matplotlib.axes._subplots.AxesSubplot): matplotlib axes object

    """
    fig, ax = plt.subplots()
    ax.plot(t_p, current_p, "--", label="Pearson", color="black")
    ax.plot(t_f, current_f, label="FOCS", color="royalblue")
    ax.fill_between(
        t_f,
        current_f - sig_current_f,
        current_f + sig_current_f,
        color="royalblue",
        alpha=0.5,
        label=r"$\pm 1 \sigma{}$",
    )
    ax.set_xlabel(r"Time [$\mu$s]")
    ax.set_ylabel(r"$I (t)$ [A]")
    ax.legend()
    return fig, ax


def current_plot_just_F(t_f, current_f):
    """Generate current plot.

    Args:
        t_f (np.ndarray): time data for Faraday
        current_f (np.ndarray): current data for Faraday

    Returns:
        fig (matplotlib.figure.Figure): matplotlib figure object
        ax (matplotlib.axes._subplots.AxesSubplot): matplotlib axes object

    """
    fig, ax = plt.subplots()
    ax.plot(t_f, current_f, label="Faraday")
    ax.set_xlabel(r"Time [$\mu$s]")
    ax.set_ylabel(r"$I (t)$")
    ax.legend()
    return fig, ax


def generate_Lissajous_animation(time, x, y, title, samples):
    """Generate a gif animation of a Lissajous plot.

    Args:
        time (np.ndarray): time array
        x (np.ndarray): x array
        y (np.ndarray): y array
        title (string): plot title
        samples (int): samples for gif

    Returns:
        None.

    """
    frames = []
    cs = resample_cs(time, x, y)
    ts = np.linspace(time[0], time[-1], samples)
    (
        frx,
        fry,
    ) = (
        cs(ts)[:, 0],
        cs(ts)[:, 1],
    )
    for i, t in enumerate(ts):
        fig, ax = plt.subplots()
        t = np.round(t, 5)
        ax.set_title(rf"{title} - $t = {t} \mu S$")
        ax.plot(frx, fry, "--", alpha=0.5, color="black")
        ax.plot(frx[0:i], fry[0:i], "-", alpha=0.75, color="black")
        ax.plot(frx[i], fry[i], ".", ms=5, color="black")
        ax.set_xlabel("S1")
        ax.set_ylabel("S2")
        ax.grid()
        ax.set_box_aspect(1)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        frames.append(image.reshape(*reversed(fig.canvas.get_width_height()), 3))
    imageio.mimsave(
        f"./animation_{title}.gif",  # output gif
        frames,  # array of input frames
        fps=20,
    )  # optional: frames per second


def point_heat_map(x, y, ax):
    """
    Generate heat map based on number of points in Lissajous figure.

    Args:
        x (np.ndarray): x data
        y (np.ndarray): y data
        ax (matplotlib.axes._subplots.AxesSubplot): matplotlib axes object

    Returns:
        None.

    """
    Z, xedges, yedges = np.histogram2d(x, y, bins=50)
    Z = np.ma.masked_array(Z, Z < 1)
    ax.pcolormesh(xedges, yedges, Z.T, cmap="viridis")
    return None


def signals_post_filter(time, x, y, x_f, y_f):
    fig, ax = plt.subplots(2, 1)

    # fig.suptitle('Filtering Comparison')

    ax[0].plot(time, x - x.mean(), "-", label="Ch.3 raw", color="black")
    ax[1].plot(time, y - y.mean(), "-", label="Ch.4 raw", color="black")

    ax[0].plot(
        time,
        x_f,
        "--",
        label="Ch.3 filtered",
        alpha=0.75,
        color="red",
    )
    ax[1].plot(
        time,
        y_f,
        "--",
        label="Ch.4 filtered",
        alpha=0.75,
        color="red",
    )

    ax[1].set_xlabel("Time (microseconds)")
    ax[1].set_ylabel("Signal (A.U.)")

    ax[0].set_xlabel("Time (microseconds)")
    ax[0].set_ylabel("Signal (A.U.)")

    ax[0].legend()
    ax[1].legend()

    return fig, ax
