# Robot Learning - Group 169
# Victor Jimenez 2491031
# Daniel Piendl 2586991
# Dominik Marino 2468378
#
# CTL is the name of the controller.
# Q_HISTORY is a matrix containing all the past position of the robot. Each row of this matrix is [q_1, ... q_i], where
# i is the number of the joints.
# Q and QD are the current position and velocity, respectively.
# Q_DES, QD_DES, QDD_DES are the desired position, velocity and acceleration, respectively.
# GRAVITY is the gravity vector g(q).
# CORIOLIS is the Coriolis force vector c(q, qd).
# M is the mass matrix M(q).

import numpy as np

gainMultiplier = 1

kp = np.array((60, 30)) * gainMultiplier
kd = np.array((10, 6)) * gainMultiplier
ki = np.array((0.1, 0.1)) * gainMultiplier


def my_ctl(ctl, q, qd, q_des, qd_des, qdd_des, q_hist, q_deshist, gravity, coriolis, M):
    if ctl == 'P':
        u = kp * (q_des - q)
    elif ctl == 'PD':
        u = kp * (q_des - q) + kd * (qd_des - qd)
    elif ctl == 'PID':
        if q_hist.shape[0] == 0:
            u = kp * (q_des - q) + kd * (qd_des - qd)
        else:
            u = kp * (q_des - q) + kd * (qd_des - qd) + ki * np.sum(q_deshist - q_hist, axis=0)
    elif ctl == 'PD_Grav':
        u = kp * (q_des - q) + kd * (qd_des - qd) + gravity
    elif ctl == 'ModelBased':
        u = M @ (qdd_des + kp * (q_des - q) + kd * (qd_des - qd)) + coriolis + gravity
    else:
        raise Warning(f"wrong definition for ctl: {ctl}")
    u = np.mat(u).T
    return u
