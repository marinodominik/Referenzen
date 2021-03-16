import numpy as np
import matplotlib.pyplot as plt


def getProMPBasis(dt, nSteps, n_of_basis, bandwidth):
    nBasis = n_of_basis
    time = np.arange(dt, nSteps * dt, dt)
    assert nSteps == len(time)
    Ts = nSteps * dt - dt

    C = np.zeros(nBasis)  # Basis function centres
    H = np.zeros(nBasis)  # Basis function bandwidths

    # for i in range(nBasis):
    #     C[i] = -2 * bandwidth + (Ts + 4 * bandwidth) / nBasis * i
    C = np.linspace(0 - 2 * bandwidth, Ts + 2 * bandwidth, nBasis)
    # for i in range(nBasis):
    #     H[i] = bandwidth ** 2

    Phi = np.zeros((nBasis, nSteps))

    for k, time_k in enumerate(time):
        for j in range(nBasis):
            Phi[j, k] = np.exp(-.5 * (time_k - C[j]) ** 2 / bandwidth ** 2)  # Basis function activation over time
    for k in range(Phi.shape[1]):
        Phi[:, k] = (Phi[:, k]) / np.sum(Phi[:, k])  # Normalize basis functions and weight by canonical state

    return Phi


if __name__ == '__main__':
    dt = 0.002
    nBasis = 30
    bandwidth = 0.2
    time = np.arange(-2*bandwidth, 3+2*bandwidth, dt)
    nSteps = len(time)
    Phi = getProMPBasis(dt, nSteps, nBasis, bandwidth)

    fig0, ax0 = plt.subplots(1)
    for ii in range(Phi.shape[0]):
        ax0.plot(time, Phi[ii])
    # ax0.plot(time, sum(Phi[i] for i in range(Phi.shape[0])))

    plt.xlim(-2*bandwidth, 3+2*bandwidth)
    plt.grid()
    plt.savefig("ProMP_basis_function.pdf")
    plt.show()
