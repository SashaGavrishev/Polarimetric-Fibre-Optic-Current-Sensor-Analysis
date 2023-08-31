# -*- coding: utf-8 -*-
"""Fast Levenberg Marquardt Step"""

import numpy as np


def levenberg_marquardt_step(struct, rho):
    jacobian_matrix = struct["jac_mat"]
    r = struct["r"]
    lambda_l = struct["lambda"]
    delta = struct["delta"][:, struct["k"]]
    damping_multiplier = struct["damping_multiplier"]
    damping_divisor = struct["damping_divisor"]
    current_cost = struct["cost"][0][struct["k"]]
    data_points = struct["data_points"]
    covList = struct["covList"]
    numberOfPoints = struct["numberOfPoints"]
    H = struct["H"]
    jlp = struct["jacob_latentParameters"]
    eta = struct["eta"][:, struct["k"]]

    t = (
        np.array(
            [
                1,
                2 * eta[0],
                eta[0] ** 2 + np.abs(eta[1]) ** rho,
                eta[2],
                eta[3],
                eta[4],
            ]
        )
        .conj()
        .T
    )

    t = t / np.linalg.norm(t)

    jacob = jacobian_matrix.conj().T @ r

    DMP = (jlp.conj().T @ jlp) * lambda_l
    update_a = -np.linalg.lstsq(H + DMP, jacob, rcond=None)[0]

    DMP = (jlp.conj().T @ jlp) * lambda_l / damping_divisor
    update_b = -np.linalg.lstsq(H + DMP, jacob, rcond=None)[0]

    eta_potential_a = eta + update_a.T.flatten()
    eta_potential_b = eta + update_b.T.flatten()

    t_potential_a = (
        np.array(
            [
                1,
                2 * eta_potential_a[0],
                eta_potential_a[0] ** 2
                + np.abs(eta_potential_a[1]) ** rho,
                eta_potential_a[2],
                eta_potential_a[3],
                eta_potential_a[4],
            ]
        )
        .conj()
        .T
    )

    t_potential_a = t_potential_a / np.linalg.norm(t_potential_a)

    t_potential_b = (
        np.array(
            [
                1,
                2 * eta_potential_b[0],
                eta_potential_b[0] ** 2
                + np.abs(eta_potential_b[1]) ** rho,
                eta_potential_b[2],
                eta_potential_b[3],
                eta_potential_b[4],
            ]
        )
        .conj()
        .T
    )

    t_potential_b = t_potential_b / np.linalg.norm(t_potential_b)

    cost_a = 0

    cost_b = 0

    for i in range(0, numberOfPoints):
        m = data_points[:, i]

        ux_i = (
            np.array(
                [[m[0] ** 2, m[0] * m[1], m[1] ** 2, m[0], m[1], 1]]
            )
            .conj()
            .T
        )

        dux_i = (
            np.array(
                [
                    [2 * m[0], m[1], 0, 1, 0, 0],
                    [0, m[0], 2 * m[1], 0, 1, 0],
                ]
            )
            .conj()
            .T
        )

        A = ux_i @ ux_i.conj().T

        covX_i = covList[i]

        B = dux_i @ covX_i @ dux_i.conj().T

        t_aBt_a = t_potential_a.conj().T @ B @ t_potential_a
        t_aAt_a = t_potential_a.conj().T @ A @ t_potential_a

        t_bBt_b = t_potential_b.conj().T @ B @ t_potential_b
        t_bAt_b = t_potential_b.conj().T @ A @ t_potential_b

        cost_a = cost_a + np.abs(t_aAt_a / t_aBt_a)
        cost_b = cost_b + np.abs(t_bAt_b / t_bBt_b)

    if (cost_a >= current_cost) and (cost_b >= current_cost):
        struct["eta_updated"] = False
        struct["cost"][0][struct["k"] + 1] = current_cost
        struct["eta"][:, struct["k"] + 1] = eta
        struct["t"][:, struct["k"] + 1] = t
        struct["delta"][:, struct["k"] + 1] = delta
        struct["lambda_l"] = lambda_l * damping_multiplier

    elif cost_b < current_cost:
        struct["eta_updated"] = True
        struct["cost"][0][struct["k"] + 1] = cost_b
        struct["eta"][:, struct["k"] + 1] = eta_potential_b
        struct["t"][:, struct["k"] + 1] = t_potential_b
        struct["delta"][:, struct["k"] + 1] = update_b.conj().T
        struct["lambda_l"] = lambda_l / damping_divisor
    else:
        struct["eta_updated"] = True
        struct["cost"][0][struct["k"] + 1] = cost_a
        struct["eta"][:, struct["k"] + 1] = eta_potential_a
        struct["t"][:, struct["k"] + 1] = t_potential_a
        struct["delta"][:, struct["k"] + 1] = update_a.conj().T
        struct["lambda_l"] = lambda_l

    return struct
