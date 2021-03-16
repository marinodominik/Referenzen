# Robot Learning - Group 169
# Victor Jimenez 2491031
# Daniel Piendl 2586991
# Dominik Marino 2468378
#
# This is one of the two main classes you need to run.
#
# CTLS is a list of cells with your controller name
# ('JacTrans','JacPseudo,'JacDPseudo','JacNullSpace'). You can run more controllers
# one after another, e.g. passing {'JacTrans','JacPseudo'}.
#
# PAUSETIME is the number of seconds between each iteration for the
# animated plot. If 0 only the final position of the robot will be
# displayed.

import numpy as np
from simSys import *
from DoubleLink import *


def taskCtlComp(ctls=['JacDPseudo'], pauseTime=False, resting_pos=None):
    dt = 0.002
    robot = DoubleLink()
    robot.friction = np.array([2.5, 2.5])
    t_end = 3.0
    time = np.arange(0, t_end, dt)
    nSteps = len(time)
    numContrlComp = len(ctls)
    target = {}
    target['x'] = np.tile([-0.35, 1.5], (nSteps, 1))
    target['cartCtl'] = True
    states = simSys(robot, dt, nSteps, ctls, target, pauseTime, resting_pos)

    taskSpace_plot(states, robot, ctls)


def taskSpace_plot(states, robot: DoubleLink, ctls):
    fig = plt.figure()
    ax1 = fig.add_subplot(111, autoscale_on=False, xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
    ax1.plot(2 * np.array([-1.1, 1.1]), np.array([0, 0]), 'b--')
    line1, = ax1.plot([], [], 'o-', lw=2, color='k', markerfacecolor='w', markersize=12)
    ax1.set_xlabel('x-axis in m', fontsize=15)
    ax1.set_ylabel('y-axis in m', fontsize=15)
    ax1.set_title('Initial Position')
    robot.visualize(states[0, :], line1)
    ax1.scatter([-.35], [1.5], marker='x', c='red',zorder=10, label='Setpoint')
    ax1.legend()
    plt.title('Initial Position for Task Space Control')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
    # Uncomment to save
    # fig.savefig(fname="SavedPlots/" + "TaskCtlInitial_" + ".pdf", format='pdf')

    fig = plt.figure()
    ax2 = fig.add_subplot(111, autoscale_on=False, xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
    ax2.plot(2 * np.array([-1.1, 1.1]), np.array([0, 0]), 'b--')
    line2, = ax2.plot([], [], 'o-', lw=2, color='k', markerfacecolor='w', markersize=12)
    ax2.set_xlabel('x-axis in m', fontsize=15)
    ax2.set_ylabel('y-axis in m', fontsize=15)
    ax2.set_title('Final Position after 3s')
    robot.visualize(states[-1, :], line2)
    ax2.scatter([-.35], [1.5], marker='x', c='red',zorder=10, label='Setpoint')
    ax2.legend()
    plt.title(f'Final Position for {str(ctls[0])} Control')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
    # Uncomment to save
    # fig.savefig(fname="SavedPlots/" + "TaskCtl_" + str(ctls[0]) + ".pdf", format='pdf')
