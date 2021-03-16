import numpy as np
import matplotlib.pyplot as plt
from getImitationData import *
from getProMPBasis import *


def proMP(nBasis, condition=False):
    dt = 0.002
    time = np.arange(dt, 3, dt)
    nSteps = len(time)
    data = getImitationData(dt, time, multiple_demos=True)
    q = data[0]
    qd = data[1]
    qdd = data[2]

    bandwidth = 0.2
    Phi = getProMPBasis(dt, nSteps, nBasis, bandwidth)

    sigma = 10 ** -12

    # https://www.ias.informatik.tu-darmstadt.de/uploads/Team/AlexandrosParaschos/promps_auro.pdf
    # Eq. 13
    w = np.linalg.inv(Phi @ Phi.transpose() + sigma ** 2 * np.identity(Phi.shape[0])) @ Phi @ q.transpose()
    # ToDo: Hier wirklich q verwenden?

    mean_w = np.mean(w, axis=1)
    cov_w = np.cov(w, rowvar=True)
    mean_traj = np.mean(q, axis=0)
    std_traj = np.std(q, axis=0)

    if not condition:
        plt.figure()
        plt.fill_between(time, mean_traj - 2 * std_traj, mean_traj + 2 * std_traj, alpha=0.5, edgecolor='#1B2ACC',
                         facecolor='#089FFF')

        # Plot all trajectories
        plt.plot(time, q.T, alpha=.5)

        plt.plot(time, mean_traj, color='#1B2ACC', label='Mean observed trajectory', linewidth=2)
        # Plot learned trajectory
        plt.plot(time, Phi.transpose() @ mean_w, label="Imitated trajectory", linewidth=2, color='black')

        title = 'ProMP with ' + str(nBasis) + ' basis functions'
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.xlim((0, 3))

    # Conditioning
    else:
        y_d = 3
        Sig_d = 0.0002
        t_point = np.int(2300 / 2)

        tmp = np.dot(cov_w, Phi[:, t_point]) / (Sig_d + np.dot(Phi[:, t_point].T, np.dot(cov_w, Phi[:, t_point])))

        cov_w_new = cov_w - tmp.reshape((30, 1)) @ (Phi[:, t_point].dot(cov_w)).reshape((1, 30))
        mean_w_new = mean_w + tmp * (y_d - Phi[:, t_point].T.dot(mean_w))
        mean_traj_new = Phi.T @ mean_w_new
        std_traj_new = np.sqrt(np.diagonal(sigma * np.eye(Phi.shape[1]) + Phi.T @ cov_w_new @ Phi))

        sample_traj = np.dot(Phi.T, np.random.multivariate_normal(mean_w_new, cov_w_new, 10).T)

        plt.figure()
        plt.fill_between(time, mean_traj - 2 * std_traj, mean_traj + 2 * std_traj, alpha=0.5, edgecolor='#1B2ACC',
                         facecolor='#089FFF')
        plt.fill_between(time, mean_traj_new - 2 * std_traj_new, mean_traj_new + 2 * std_traj_new, alpha=0.5,
                         edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.plot(time, sample_traj, alpha=.5)
        plt.plot(time, mean_traj, color='#1B2ACC', linewidth=2, label='Mean old trajectory')
        plt.plot(time, mean_traj_new, label='Mean new trajectory', linewidth=2, color='black')

        plt.scatter(t_point * dt, y_d, label="Via point")

        plt.xlim((0, 3))
        title = 'ProMP after conditioning with new sampled trajectories'
        plt.title(title)
        plt.grid()
        plt.legend()

    plt.savefig(title + ".pdf")
    plt.draw_all()
    plt.pause(0.001)
