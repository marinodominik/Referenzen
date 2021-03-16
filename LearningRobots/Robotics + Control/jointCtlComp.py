# Robot Learning - Group 169
# Victor Jimenez 2491031
# Daniel Piendl 2586991
# Dominik Marino 2468378
#
# This is one of the two main classes you need to run.
#
# CTLS is a list of cells with your controller name
# ('P','PD,'PID','PD_Grav','ModelBased'). You can run more controllers
# one after another, e.g. passing {'P','PD_Grav'}.
#
# ISSETPOINT is a boolean variable. If 1 the robot has to reach a fixed
# point, if 0 it will follow a fixed trajectory.
#
# PAUSETIME is the number of seconds between each iteration for the
# animated plot. If 0 only the final position of the robot will be
# displayed.

import numpy as np
from math import pi, sin, cos
from simSys import *
from DoubleLink import *
from matplotlib import pyplot as plt


def jointCtlComp(ctls=['P'], isSetPoint=False, pauseTime=False):
    dt = 0.002
    robot = DoubleLink()
    robot.friction = np.array([2.5, 2.5])
    t_end = 3.0
    time = np.arange(0, t_end, dt)
    nSteps = len(time)
    numContrlComp = len(ctls)
    target = {'cartCtl': False}
    if isSetPoint:
        target['q'] = np.tile([-pi / 2, 0], (nSteps, 1))
        target['qd'] = np.tile([0, 0], (nSteps, 1))
        target['qdd'] = np.tile([0, 0], (nSteps, 1))
    else:
        f1 = 2
        f2 = 0.5
        target['q'] = np.array([np.sin(2 * pi * f1 * time) - pi, np.sin(2 * pi * f2 * time)]).T
        target['qd'] = np.array(
            [2 * pi * f1 * np.cos(2 * pi * f1 * time), 2 * pi * f2 * np.cos(2 * pi * f2 * time)]).T
        target['qdd'] = np.array([-(2 * pi * f1) ** 2 * np.sin(2 * pi * f1 * time),
                                  -(2 * pi * f2) ** 2 * np.sin(2 * pi * f2 * time)]).T

    states = simSys(robot, dt, nSteps, ctls, target, pauseTime)
    traj_plot(states, numContrlComp, ctls, target['q'], target['qd'], time, 0)
    traj_plot(states, numContrlComp, ctls, target['q'], target['qd'], time, 1)
    plt.pause(0.001)


# Just a way to plot, feel free to modify!
def traj_plot(states, numContrlComp, ctls, q_desired, qd_desired, time, plotVel):
    stateNo = (1, 2)
    linestyle = ((0, (5, 1)), (0, (3, 1, 1, 1)), 'dashed', '-.',  'dotted',)
    colors = ('red', 'lightblue', 'green', 'blue', 'orange')

    tracked = True
    if q_desired[0, 0] == q_desired[1, 0]:
        tracked = False

    for statei in stateNo:

        if plotVel:
            y = qd_desired
        else:
            y = q_desired

        plt.figure()

        names = list()
        plt.plot(time, y[:, statei - 1], linewidth=2, linestyle='solid', c='black', alpha=1)
        names += ['Desired_' + str(statei)]

        for k in range(numContrlComp):
            names += [ctls[k] + '_' + str(statei)]
            if k == 0:
                plt.plot(time, states[k * len(time):(k + 1) * len(time), plotVel + 2 * (statei - 1)::4], linewidth=2,
                         alpha=.3, linestyle=linestyle[k], c=colors[k])
            else:
                plt.plot(time, states[k * len(time):(k + 1) * len(time), plotVel + 2 * (statei - 1)::4], linewidth=2,
                        alpha=.8, linestyle=linestyle[k], c=colors[k])

        plt.legend(tuple(names))
        plt.xlabel('time in s', fontsize=15)

        if plotVel:
            plt.ylabel('angular velocity in rad/s', fontsize=15)
            plt.title('Velocity for Joint ' + str(statei), fontsize=20)
        else:
            plt.ylabel('angle in rad', fontsize=15)
            plt.title('Position for Joint ' + str(statei), fontsize=20)

        plt.xlim(0, 3)
        if False:  # Set to True when using high gains
            if statei == 2:
                if plotVel:
                    plt.ylim(-10, 10)
                else:
                    plt.ylim(-2, 2)
            else:
                if plotVel:
                    plt.ylim(-15, 15)
                else:
                    plt.ylim(-5, 0)
        plt.grid()
        plt.show()

        # Uncomment to save
        #plt.savefig(fname="SavedPlots/" + "Velocity" * plotVel + "Position" * (1 - plotVel) + "_Joint_" + str(statei)
        #                  + "_tracked" * tracked + ".pdf", format='pdf')  # + "_HighGains"
