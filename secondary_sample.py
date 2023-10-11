"""
Code Sample with a Better Error Analysis
If using matlab code of FGEE is preferred, the code for linking matlab is given in the sample
Be sure to add path to this folder in matlab path environment
"""

import analysis_tools.filtering as filtering
import matplotlib.pyplot as plt
import matlab.engine
import numpy as np
import os
import json
import scipy as sp
import scipy.optimize
import scipy.interpolate
import pandas as pd
import re
from ellipse_fitting.fast_guaranteed_estimate import fgee_estimate
from ellipse_fitting.direct_fit import direct_ellipse_fit
from ellipse_fitting.cov_geo_param import cov_geo_params
from ellipse_fitting.ellipse_tools import ellipse_parameters


###################
# Analysis Class
###################

class NewDataAnalysis(object):
    def __init__(self,Time, CH3, CH4, CH1=[], dataName = "Untitled", factor = 1):
        self.Time = Time
        self.CH1 = CH1
        self.CH3 = CH3
        self.CH4 = CH4
        self.CH3_hist = []
        self.CH4_hist = []
        self.start = 0
        self.end = len(CH3)
        self.signal_end = len(CH3)
        self.remap_new, self.remap_std = [[],[]], [[],[]]
        self.geo_cov, self.geo_param = [], []
        self.dataName = dataName
        self.theta = []
        self.theta_std = []
        self.factor = factor
        self.result = []
        self.theta_0 = 0
        self.sx = [0]
        self.sy = [0]

    def filter(self, level=7, save=True):
        CH4_Filter = filtering.filter_sig(self.Time, self.CH4)
        CH4_Filter.rm_DC_offset()
        CH4_Filter.apply_dwt(level)
        _, CH4_V = CH4_Filter.signal()


        CH3_Filter = filtering.filter_sig(self.Time, self.CH3)
        CH3_Filter.rm_DC_offset()
        CH3_Filter.apply_dwt(level)
        _, CH3_V = CH3_Filter.signal()
        if save == True:
            self.CH3 = CH3_V
            self.CH4 = CH4_V

    def twoD_hist(self, bins=500):
        """Resample 2D data using a histogram."""
        Z, xedges, yedges = np.histogram2d(self.CH3[self.start:self.end], self.CH4[self.start:self.end], bins=bins)

        x_c = np.convolve(xedges, np.ones(2) / 2, mode="valid")
        y_c = np.convolve(yedges, np.ones(2) / 2, mode="valid")

        Pts = np.vstack((x_c, y_c))

        argZ = np.argwhere(Z != 0).T

        x_c, y_c = Pts[[[0], [1]], argZ]

        self.CH3_hist = x_c
        self.CH4_hist = y_c
        return x_c, y_c

    def plot_graph(self, text):
        text_list = ["analysed data", "raw Lissajous","remapped Lissajous"]
        if text not in range(len(text_list)):
            raise(f"Error: argument of plot_graph() should be integer in "
                  f"range of 0 and {int(len(text_list)-1)}")
            return None
        text = text_list[text]

        if text == "analysed data" and len(self.CH1) != 0:
            plt.plot(self.Time, self.CH1, label='original data')
            plt.plot(self.Time[self.start:self.signal_end], self.CH1[self.start:self.signal_end],
                     label='data of interest')
            plt.plot(self.Time[self.start:self.end], self.CH1[self.start:self.end],
                     label='ellipse fitting data')
            plt.title("Data Used for Analysis")
            plt.legend()
            plt.show()

        elif text == "raw Lissajous" and len(self.remap_new[0]) != 0:
            x, y = plot_ellipse(*self.geo_param, 1000)
            plt.plot(x, y, label='new fit')
            plt.plot(self.CH3, self.CH4, label="filtered raw data")
            if len(self.CH3_hist) != 0:
                plt.plot(self.CH3_hist, self.CH4_hist, ".", label="histogramed fitted data", alpha=0.5)
            plt.legend()
            plt.title(f'Lissajous Figure - Raw :\n {self.dataName}')
            plt.show()

        elif text == "remapped Lissajous" and len(self.remap_std[0]) != 0:
            fig, ax = plt.subplots()
            ax.plot(self.remap_new[0], self.remap_new[1])
            ax.set_box_aspect(1)
            ax.set_title('Lissajous Figure')
            ax.set_xlabel(r'Signal 1 [A.U.]')
            ax.set_ylabel(r'Signal 2 [A.U.]')
            fig.tight_layout()
            ax.set_title(f'Lissajous Figure - Remapped :\n {self.dataName}')
            theta = np.linspace(0, 2 * np.pi, 1000)
            x_c, y_c = np.cos(theta), np.sin(theta)
            ax.plot(x_c, y_c, '--', label='unit circle')

            ax.errorbar(self.remap_new[0], self.remap_new[1],
                        xerr=self.remap_std[0], yerr=self.remap_std[1], ecolor="k")
            '''
            ax.fill_between(self.remap_new[0], self.remap_new[1]+self.remap_std[1],
                            self.remap_new[1]-self.remap_std[1], color="k", alpha=0.3)
            ax.fill_betweenx(self.remap_new[1], self.remap_new[0] + self.remap_std[0],
                            self.remap_new[0] - self.remap_std[0], color="k", alpha=0.3)
            '''
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            plt.show()


    def find_geo_param(self):
        ### When using matlab code instead of python code of FGEE
        """
        if os.path.isfile("/Users/emoon/PycharmProjects/UROP/EfieldSensor/myfile.json"):
            out_file = open("/Users/emoon/PycharmProjects/UROP/EfieldSensor/myfile.json", "w")
        else:
            out_file = open("/Users/emoon/PycharmProjects/UROP/EfieldSensor/myfile.json", "a")

        start = self.start
        end = self.end

        if len(self.CH1) == 0:
            json.dump({"Time": list(self.Time[start:end]),
                       "CH3_V": list(self.CH3[start:end]), "CH4_V": list(self.CH4[start:end])}, out_file)
        elif len(self.CH3_hist) == 0:
            json.dump({"Time": list(self.Time[start:end]), "CH1_V": list(self.CH1[start:end]),
                       "CH3_V": list(self.CH3[start:end]), "CH4_V": list(self.CH4[start:end])}, out_file)
        else:
            json.dump({"Time": list(self.Time[start:end]), "CH1_V": list(self.CH1[start:end]),
                       "CH3_V": list(self.CH3_hist), "CH4_V": list(self.CH4_hist)}, out_file)

        out_file.close()

        eng = matlab.engine.start_matlab()
        result = eng.Efield_get_ellipse_from_matLab()
        geo_param = [i[0] for i in result]
        geo_cov = [i[1] for i in result]
        self.geo_param = geo_param[:2] + [-geo_param[-1]] + [np.array(geo_param[2:4])]
        self.geo_cov = geo_cov[:2] + [geo_cov[-1]] + [np.array(geo_cov[2:4])]
        """

        ### When using python code for FGEE
        theta_fgee = fgee_estimate(self.CH3[start:end], self.CH4[start:end], direct_ellipse_fit(self.CH3[start:end], self.CH4[start:end]))
        geoCov = cov_geo_params(np.array([theta_fgee]).T, self.CH3[start:end], self.CH4[start:end])

        self.geo_cov = np.sqrt(np.diag(geoCov))
        self.geo_param = ellipse_parameters(theta_fgee)

        return self.geo_param, self.geo_cov

    def remap_ellipse(self):
        if len(self.geo_param) == 0 or len(self.geo_cov) == 0:
            raise (f"Error: self.geo_param or self.geo_cov is not defined\nself.geo_param={self.geo_param}"
                   f"self.geo_cov={self.geo_cov}")
            return None

        self.remap_new , self.remap_std = map_ellipse_with_error(*self.geo_param, *self.geo_cov,
                               self.CH3[self.start:self.signal_end],
                               self.CH4[self.start:self.signal_end], self.sx, self.sy)

        self.Time = self.Time[self.start:self.signal_end]
        return self.remap_new, self.remap_std

    def find_theta(self):

        if len(self.remap_new[0]) == 0 or len(self.remap_std[0]) == 0:
            raise (f"Error: self.remap_new or self.remap_std is not defined")
            return None

        x_remap = self.remap_new[0]
        y_remap = self.remap_new[1]
        x_std = self.remap_std[0]
        y_std = self.remap_std[1]

        theta_0 = np.arctan2(x_remap[0], y_remap[0])
        self.theta_0 = theta_0
        theta = [0]
        last = 0

        for i, t in enumerate(self.Time):
            if i != 0:
                try:
                    theta_t = (np.arctan2(x_remap[i], y_remap[i]))
                    theta_t = (theta_t-last + np.pi) % (2*np.pi)-np.pi + last
                    last = theta_t
                    theta.append(0.5*theta_t - 0.5*theta_0)
                except:
                    print(x_remap, y_remap)
                    break

        df_dx = 1/(1+ (x_remap/y_remap)**2) * 1/y_remap
        df_dy = 1/(1+ (x_remap/y_remap)**2) * (-x_remap/y_remap**2)
        theta_std = np.sqrt(np.square(df_dx) * np.square(x_std) + np.square(df_dy) * np.square(y_std))
        theta_std = 0.5 * theta_std

        self.theta = np.array(theta) * 2 # factor of 2 coming from the equation
        self.theta_std = np.array(theta_std) * 2

        return self.Time, self.theta, self.theta_std

    def _process(self):
        self.find_geo_param()
        self.remap_ellipse()
        self.find_theta()
        return self.theta, self.theta_std

    def process_factor(self):
        if len(self.CH1) == 0:
            raise("Error: self.CH1 not defined")
            return None

        if len(self.theta) == 0:
            self._process()

        cs = sp.interpolate.CubicSpline(self.Time, self.CH1)

        def function(t, a, b):
            return (cs(t - b)) * a

        popt, pcov = sp.optimize.curve_fit(function, self.Time, self.theta, p0=[1, 0])
        self.factor *= popt[0]

        popt_min, pcov_min = sp.optimize.curve_fit(function, self.Time, self.theta + self.theta_std, p0=[1, 0])
        popt_max, pcov_max = sp.optimize.curve_fit(function, self.Time, self.theta - self.theta_std, p0=[1, 0])

        print(f"Proportionality Constant: {popt[0]}\nUncertainty: {np.sqrt(pcov[0][0])}")
        print(f"Min Proportionality Constant: {popt_min[0]}\nUncertainty: {np.sqrt(pcov_min[0][0])}")
        print(f"Max Proportionality Constant: {popt_max[0]}\nUncertainty: {np.sqrt(pcov_max[0][0])}")
        print(f"Shifting Constant: {popt[1]}\nUncertainty: {np.sqrt(pcov[1][1])}")

        fig, ax = plt.subplots()
        ax.plot(self.Time - popt[1], self.theta/popt[0], label='PockelsCell')
        ax.plot(self.Time, self.CH1, label='E Field', alpha=0.3)
        ax.set_xlabel(r'Time [$\mu$s]')
        ax.set_ylabel(r'$I (t)$')
        ax.legend(loc="lower right")
        ax.fill_between(self.Time - popt[1], (self.theta + self.theta_std) / popt[0],
                        (self.theta - self.theta_std) / popt[0],
                         color='black', alpha=0.3)
        ax.set_title(f'Remapped Data :\n {self.dataName}')
        plt.show()

        prop_dict = {"average": [popt[0], np.sqrt(pcov[0][0])],
                     "crude_uncertainty": popt[0] * np.sqrt(np.average(self.theta_std ** 2))/np.average(np.abs(self.theta)),
                     "min": [popt_min[0], np.sqrt(pcov_min[0][0])], \
                     "max": [popt_max[0], np.sqrt(pcov_max[0][0])]}

        return prop_dict

    def process_data(self, isDebug=False):
        if len(self.theta) == 0:
            self._process()

        print(f"The Calibration Factor is {self.factor}")
        print(f"Calibration Factor is k in kE = $\Phi$")
        self.result = self.theta / self.factor

        if isDebug == True:
            fig, ax = plt.subplots()
            ax.plot(self.Time, self.result, label='PockelsCell')
            ax.set_xlabel(r'Time [$\mu$s]')
            ax.set_ylabel(r'$V (mV)$')
            ax.legend(loc="lower right")
            ax.set_title(f'Processed Data :\n {self.dataName}')
            plt.show()

########################
# Functions used
########################
def map_ellipse_with_error(alpha, beta, phi, cent, s_alpha, s_beta, s_phi, s_cent,x, y,sx=0, sy=0):
    """
    Use an ellipse's geometric parameters, to remap ellipse to a unit circle.

    Given ellipse data points X = [x_, y_]' where x_ and y_ are 1D arrays,
    can always offset, then rotate, then scale an ellipse to reduce it to
    a unit circle.

    Args:
        alpha (float): alpha scaling factor
        beta (float): beta scaling factor
        phi (float): rotation angle [rad]
        cent (np.ndarray): centre vector of ellipse
        x : ellipse x-data
        y (np.ndarray): ellipse y-data

    Returns:
        X (np.ndarray): remapped ellipse points to a unit circle

    """
    X = np.array([x, y])
    sx = np.array([sx, [0]])
    sy = np.array([[0], sy])

    cent = np.array([[cent[0]], [cent[1]]])

    R_phi = np.array([[np.cos(phi), - np.sin(phi)],
                     [np.sin(phi), np.cos(phi)]])

    dR_phi_dphi = np.array([[-np.sin(phi), - np.cos(phi)],
                     [np.cos(phi), -np.sin(phi)]])
    scale = np.array([[1/alpha, 0], [0, 1/beta]])

    result = (scale @ R_phi @ (X - cent))
    df_dx = (scale @ R_phi @ sx)
    df_dy = (scale @ R_phi @ sy)

    df_dphi = scale @ dR_phi_dphi @ (X-cent)

    df_dx0 = scale @ R_phi @ (-1 * np.array([[cent[0][0]], [0]]))
    df_dy0 = scale @ R_phi @ (-1 * np.array([[0], [cent[1][0]]]))

    df_dscale = -1 * np.array([[1/alpha, 0], [0, 1/beta]]) @ result

    print(df_dx, df_dy)
    s_scale = np.array([[s_alpha],[s_beta]])
    error = np.square(df_dphi) * s_phi **2 + np.square(df_dx0) * s_cent[0] **2 + \
            np.square(df_dy0) * s_cent[1] ** 2 + np.square(df_dscale) * s_scale ** 2 + \
            np.square(df_dx) + np.square(df_dy)
    std = np.sqrt(error)

    return result, std


def plot_ellipse(alpha, beta, phi, cent, samples):
    """
    Use the geometric parameters to generate an ellipse.

    Starting with a circle, perform inverse transformations to map a unit
    circle to an ellipse. Unit circle -> Scale -> Rotate -> Offset

    Args:
        alpha (float): alpha scaling factor
        beta (float): beta scaling factor
        phi (float): rotation angle [rad]
        cent (np.ndarray): centre vector of ellipse
        samples (int): number of points for ellipse

    Returns:
        X (np.ndarray): ellipse points

    """
    cent = np.array([[cent[0]], [cent[1]]])

    R_phi = np.array([[np.cos(phi), - np.sin(phi)],
                     [np.sin(phi), np.cos(phi)]])

    scale = np.array([[1/alpha, 0], [0, 1/beta]])

    theta = np.linspace(0, 2*np.pi, samples)

    X = np.array([np.cos(theta), np.sin(theta)])

    return (np.linalg.inv(R_phi) @ np.linalg.inv(scale) @ X) + cent

def calc_residual_systematic_err(x_remap, y_remap):
    """
    Initial Error propagation method (quick error propagation)
    :param x_remap: ndarray or list
    :param y_remap: ndarray or list
    :return sig_theta_systematic: ndarray
    """
    vector = np.array([x_remap, y_remap]).T
    length = np.linalg.norm(vector, axis=1)
    residual_error = np.abs(length - 1)
    sig_theta_systematic = np.arctan2(
        np.sqrt(1 - (1 - (residual_error**2) / 2) ** 2),
        (1 - (residual_error**2) / 2),
    )
    return sig_theta_systematic

def function3(x):
    global processed
    alpha, beta, phi, cent_x, cent_y = x
    cent = [cent_x, cent_y]
    x = np.cos(-processed + processed[0] + np.pi / 2 - Analysis.theta_0)
    y = np.sin(-processed + processed[0] + np.pi / 2 - Analysis.theta_0)

    cent = np.array([[cent[0]], [cent[1]]])
    scale = np.linalg.inv(np.array([[1 / alpha, 0], [0, 1 / beta]]))
    R_phi = np.linalg.inv(np.array([[np.cos(phi), - np.sin(phi)],
                                    [np.sin(phi), np.cos(phi)]]))

    processed_x = R_phi @ scale @ np.array([x, y]) + cent


    e1 = np.square((processed_x[0] - np.array(Analysis.CH3[start:signal_end])))
    e2 = np.square((processed_x[1] - np.array(Analysis.CH4[start:signal_end])))
    er2 = np.sqrt(e1 + e2)
    return [np.array([np.average(np.sqrt(e1))]), np.array([np.average(np.sqrt(e2))])]

def function2(x):
    global processed
    alpha, beta, phi, cent_x, cent_y = x
    cent = [cent_x, cent_y]
    x = np.cos(-processed + processed[0] + np.pi / 2 - Analysis.theta_0)
    y = np.sin(-processed + processed[0] + np.pi / 2 - Analysis.theta_0)

    cent = np.array([[cent[0]], [cent[1]]])
    scale = np.linalg.inv(np.array([[1 / alpha, 0], [0, 1 / beta]]))
    R_phi = np.linalg.inv(np.array([[np.cos(phi), - np.sin(phi)],
                                    [np.sin(phi), np.cos(phi)]]))

    processed_x = R_phi @ scale @ np.array([x, y]) + cent

    CH3 = Analysis.CH3[start:signal_end]
    CH4 = Analysis.CH4[start:signal_end]
    e1 = np.square((processed_x[0] - np.array(CH3)) / (max(np.array(CH3))-min(np.array(CH3))))
    e2 = np.square((processed_x[1] - np.array(np.array(CH3))) / (max(np.array(CH4))-min(np.array(CH4))))
    er2 = np.sqrt(np.sum(e1 + e2)) / len(Analysis.CH1)
    return er2

def change_parameter(x):
    temp = x[3]
    x[3] = temp[0]
    x.append(temp[1])
    return x

#########################
# Main Code
#########################

# User Defined values
path = r"data\large_proportion_of_ellipse.csv"

N_coil = 21
N_coil_error = 0.5
N_turn = 1
N_turn_error = 0.1
verdet_constant = 0.71e-6
v_error = 0.03e-6
Mirror = False

end = 50 #microSecond
signal_end = 100 #microSecond

# Set DataName
dataName = path.split("/")[-1].split(".")[0]

# Faraday Mirror
if Mirror:
    M = 2
else:
    M = 1

# Extract scope time interval
reader = pd.read_csv(path, chunksize=10, index_col=0, header=None)
chuncks = reader
df = next(chuncks)
numbers = re.findall(r'\d+', df.iloc[8].to_numpy()[1])
numbers.insert(1, '.')
scope_time_interval = float(''.join(numbers))#uS

# Read data
index, CH1_V, CH3_V, CH4_V= np.loadtxt(
    path, delimiter=',', skiprows=15, unpack=True)
time = index * scope_time_interval
scope_time_interval = 1e-4
plt.plot(CH3_V,CH4_V, ".")
plt.show()

# Slice Data
start = np.argmax(abs(CH1_V) > 1000)
end = int(end/scope_time_interval) + start
signal_end = int(signal_end/scope_time_interval) + start
print(f"These are index for the start, end of ellipse fitting, end of analysis\n"
      f"{start, end, signal_end}")

# Analysis
Analysis = NewDataAnalysis(np.array(time), np.array(CH3_V), np.array(CH4_V), CH1=CH1_V)
Analysis.start = start
Analysis.end = end
Analysis.signal_end = signal_end
Analysis.twoD_hist(5000)
Analysis.filter(7,save=True)

plt.plot(Analysis.CH3[start:end],Analysis.CH4[start:end])
plt.show()

Analysis.dataName = dataName
Analysis.process_data()
Analysis.plot_graph(1)

total_theta = -1 * Analysis.theta / verdet_constant / (2*M) / N_coil / N_turn
original_err = np.abs(total_theta * (N_coil_error/N_coil + N_turn_error/N_turn))

theta_std = calc_residual_systematic_err(Analysis.remap_new[0], Analysis.remap_new[1]) / verdet_constant / (2*M)

total_theta_std = abs(theta_std) + original_err


tck = sp.interpolate.splrep(Analysis.Time, Analysis.theta, s=10)
processed = sp.interpolate.BSpline(*tck)(Analysis.Time)

aaa = change_parameter(Analysis.geo_param[:])
geo_param = sp.optimize.minimize(function2, aaa, method='nelder-mead')
errs = function3(geo_param.x)


Analysis.sx = errs[0]
Analysis.sy = errs[1]
Analysis.Time = time
Analysis.remap_ellipse()
Analysis.find_theta()
total_theta_std2 = Analysis.theta_std / verdet_constant / (2*M) / N_coil / N_turn + original_err

plt.plot(Analysis.Time, total_theta, label="Processed Data")
plt.fill_between(Analysis.Time, total_theta - total_theta_std, total_theta + total_theta_std,
                 label="Error Min", alpha=0.3)

plt.fill_between(Analysis.Time, total_theta - total_theta_std2, total_theta + total_theta_std2,
                 label="Error", alpha=0.3)

plt.legend()
plt.show()
