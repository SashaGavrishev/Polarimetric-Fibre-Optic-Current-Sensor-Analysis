# -*- coding: utf-8 -*-
"""Fast Guaranteed Ellipse Fit"""


import numpy as np
from ellipse_fitting.levenberg_marquardt_step import (
    levenberg_marquardt_step,
)


def fgee_fit(latentParams, dPts, covList):
    eta = latentParams

    t = (
        np.array(
            [
                1,
                2 * eta[0],
                eta[0] ** 2 + np.abs(eta[1]) ** 2,
                eta[2],
                eta[3],
                eta[4],
            ]
        )
        .conj()
        .T
    )

    t = t / np.linalg.norm(t)

    struct = {}

    keep_going = True

    struct["eta_updated"] = False
    struct["lambda"] = 0.01
    struct["k"] = 0
    struct["damping_multiplier"] = 15
    struct["damping_divisor"] = 1.2
    struct["numberOfPoints"] = max(dPts.shape)
    struct["data_points"] = dPts
    struct["covList"] = covList

    maxIter = 200

    struct["tolDelta"] = 1e-7
    struct["tolCost"] = 1e-7
    struct["tolEta"] = 1e-7
    struct["tolGrad"] = 1e-7
    struct["tolBar"] = 15.5
    struct["tolDet"] = 1e-5

    Fprim = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]])
    F = np.array(
        [
            [Fprim, np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3))],
        ]
    ).reshape(1, 6, 6)[0]
    I = np.eye(6, 6)

    struct["cost"] = np.zeros((1, maxIter))
    struct["eta"] = np.zeros((5, maxIter))
    struct["t"] = np.zeros((6, maxIter))
    struct["delta"] = np.zeros((5, maxIter))

    struct["t"][:, struct["k"]] = t
    struct["eta"][:, struct["k"]] = eta

    struct["delta"][:, struct["k"]] = np.ones(5)

    while keep_going and struct["k"] < (maxIter - 1):
        struct["r"] = np.zeros((struct["numberOfPoints"], 1))

        struct["jac_mat"] = np.zeros((struct["numberOfPoints"], 5))

        eta = struct["eta"][:, struct["k"]]

        t = (
            np.array(
                [
                    [
                        1,
                        2 * eta[0],
                        eta[0] ** 2 + np.abs(eta[1]) ** 2,
                        eta[2],
                        eta[3],
                        eta[4],
                    ]
                ]
            )
            .conj()
            .T
        )

        jacob_latentParameters = np.array(
            [
                [0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0],
                [
                    2 * eta[0],
                    2 * abs(eta[1]) ** (2 - 1) * np.sign(eta[1]),
                    0,
                    0,
                    0,
                ],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        )

        Pt = np.eye(6) - (
            (t @ t.conj().T) / (np.linalg.norm(t, 2) ** 2)
        )
        jacob_latentParameters = (
            (1 / np.linalg.norm(t, 2)) * Pt @ jacob_latentParameters
        )

        t = t / np.linalg.norm(t)

        for i in range(0, struct["numberOfPoints"]):
            m = dPts[:, i]

            ux_i = (
                np.array(
                    [
                        [
                            m[0] ** 2,
                            m[0] * m[1],
                            m[1] ** 2,
                            m[0],
                            m[1],
                            1,
                        ]
                    ]
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

            A = np.outer(ux_i, ux_i)

            covX_i = covList[i]

            B = dux_i @ covX_i @ dux_i.conj().T

            tBt = t.conj().T @ B @ t
            tAt = t.conj().T @ A @ t

            struct["r"][i] = np.sqrt(np.abs(tAt / tBt))

            M = A / tBt
            Xbits = B * ((tAt) / (tBt**2))
            X = M - Xbits

            grad = (
                (
                    (X @ t)
                    / np.sqrt((np.abs(tAt / tBt) + np.spacing(1)))
                )
                .conj()
                .T
            )

            struct["jac_mat"][i, :] = grad @ jacob_latentParameters

        struct["H"] = struct["jac_mat"].conj().T @ struct["jac_mat"]
        struct["cost"][0][struct["k"]] = (
            struct["r"].conj().T @ struct["r"].conj()
        )[0][0]
        struct["jacob_latentParameters"] = jacob_latentParameters

        struct = levenberg_marquardt_step(struct, 2)

        eta = struct["eta"][:, struct["k"] + 1]

        t = (
            np.array(
                [
                    [
                        1,
                        2 * eta[0],
                        eta[0] ** 2 + np.abs(eta[1]) ** 2,
                        eta[2],
                        eta[3],
                        eta[4],
                    ]
                ]
            )
            .conj()
            .T
        )

        t = t / np.linalg.norm(t)

        tIt = t.conj().T @ I @ t
        tFt = t.conj().T @ F @ t
        barrier = (tIt / tFt)[0][0]

        M = np.array(
            [
                [t[0][0], t[1][0] / 2, t[3][0] / 2],
                [t[1][0] / 2, t[2][0], t[4][0] / 2],
                [t[3][0] / 2, t[4][0] / 2, t[5][0]],
            ]
        )

        DeterminantConic = np.linalg.det(M)

        dif_p_eta = struct["eta"][:, struct["k"] + 1]
        -struct["eta"][:, struct["k"] + 1]
        dif_m_eta = struct["eta"][:, struct["k"]]
        +struct["eta"][:, struct["k"]]

        if (
            np.min(
                [np.linalg.norm(dif_p_eta), np.linalg.norm(dif_m_eta)]
            )
            < struct["tolEta"]
            and struct["eta_updated"]
        ):
            keep_going = False
        elif (
            np.abs(struct["cost"][0][struct["k"]])
            - struct["cost"][0][struct["k"] + 1]
        ) < struct["tolCost"] and struct["eta_updated"]:
            keep_going = False
        elif (
            np.linalg.norm(struct["delta"][:, struct["k"] + 1])
        ) < struct["tolDelta"] and struct["eta_updated"]:
            keep_going = False
        elif np.linalg.norm(grad) < struct["tolGrad"]:
            keep_going = False
        elif (
            np.log(barrier) > struct["tolBar"]
            or np.abs(DeterminantConic) < struct["tolDet"]
        ):
            keep_going = False

        struct["k"] += 1

    iterations = struct["k"]
    theta = struct["t"][:, struct["k"]]
    theta = theta / np.linalg.norm(theta)

    return theta, iterations
