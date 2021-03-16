# Robot Learning - Group 169
# Victor Jimenez 2491031
# Daniel Piendl 2586991
# Dominik Marino 2468378

from my_ctl import *
from my_taskSpace_ctl import *
from matplotlib import pyplot as plt


def simSys(robot, dt, nSteps, ctls, target, pauseTime=False, resting_pos=None):
    states = np.zeros((nSteps * len(ctls), robot.dimState))
    states[:: nSteps, ::2] = np.tile([-pi, 0], (len(ctls), 1))

    plt.ion()
    if pauseTime or target['cartCtl']:
        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
        plt.plot(2 * np.array([-1.1, 1.1]), np.array([0, 0]), 'b--')
        line, = ax.plot([], [], 'o-', lw=2, color='k' ,markerfacecolor='w', markersize=12)
        plt.xlabel('x-axis [m]', fontsize=15)
        plt.ylabel('y-axis [m]', fontsize=15)

        plt.draw()
        robot.visualize(states[0, :], line)

    for k in range(len(ctls)):
        for i in range((nSteps - 1)):
            c_idx = i + k * nSteps
            gravity, coriolis, M = robot.getDynamicsMatrices(states[c_idx, :])
            if not target['cartCtl']:
                # Added k*nsteps as starting index for q_hist. Otherwise the history of the controller before is also
                # passed
                u = my_ctl(ctls[k], states[c_idx, ::2], states[c_idx, 1::2], target['q'][i, :], target['qd'][i, :],
                           target['qdd'][i, :], states[k * nSteps:c_idx, ::2], target['q'][:i, :], gravity, coriolis, M)
            else:
                J, cart = robot.getJacobian(states[c_idx, ::2], 2)
                u = my_taskSpace_ctl(ctls[k], dt, np.mat(states[c_idx, ::2]).T,
                                     np.mat(states[c_idx,1::2]).T, np.mat(gravity).T,
                                     np.mat(coriolis).T, M, np.mat(J),
                                     np.mat(cart).T,  np.mat(target['x'][i,:]).T,
                                     resting_pos)

            qdd = M ** -1 * (u - np.mat(coriolis).T - np.mat(gravity).T)
            states[c_idx + 1, 1::2] = states[c_idx, 1::2] + dt * qdd.T
            states[c_idx + 1, ::2] = states[c_idx, 0::2] + dt * states[c_idx + 1, 1::2]

            if pauseTime:
                robot.visualize(states[i + 1, :], line)
                plt.pause(pauseTime)

        if target['cartCtl']:
            robot.visualize(states[0,:], line)
            robot.visualize(states[-1,:], line)
            plt.pause(0.00001)

    return states  # Would be nice to also return u so that the control input can be plotted as well
