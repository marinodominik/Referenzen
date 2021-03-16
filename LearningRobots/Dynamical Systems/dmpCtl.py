# DMP-based controller.
#
# DMPPARAMS is the struct containing all the parameters of the DMP.
#
# PSI is the vector of basis functions.
#
# Q and QD are the current position and velocity, respectively.

import numpy as np


def dmpCtl(dmpParams, psi_i, q, qd):
    f_w = psi_i.transpose().dot(dmpParams.w)

    K_d = dmpParams.tau * dmpParams.alpha
    K_p = dmpParams.tau ** 2 * dmpParams.alpha * dmpParams.beta
    u_ff = dmpParams.tau ** 2 * f_w
    q_des = dmpParams.goal

    qdd = K_p * (q_des - q) + K_d * (-qd) + u_ff

    return qdd
