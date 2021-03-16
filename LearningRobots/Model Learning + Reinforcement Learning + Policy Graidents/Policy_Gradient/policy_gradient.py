from matplotlib import pyplot as plt
import numpy as np
from Pend2dBallThrowDMP import *
import time

# %matplotlib inline
np.set_printoptions(precision=3, linewidth=100000)

env = Pend2dBallThrowDMP()

numDim = 10
numSamples = 25
maxIter = 100
numTrials = 10


# YOUR CODE HERE

def log_gauss_derivative(x, mu, sigma):
    sigma_inv = np.diag(1 / np.diag(sigma))
    diff = x - mu
    der_mu = sigma_inv.dot(diff)
    der_sigma = 0.5 * (-sigma_inv + sigma_inv.dot(np.outer(diff, diff)).dot(sigma_inv))

    return der_mu, der_sigma


def exploration(alpha, use_baseline=True, change_sigma=False, sigma_lb=None):
    if change_sigma:
        assert sigma_lb is not None

    assert isinstance(alpha, tuple)

    fig0, ax0 = plt.subplots()


    for alpha_i in alpha:
        reward = np.zeros((numTrials, maxIter, numSamples))
        for trial_i in range(numTrials):
            mu = np.zeros(numDim)
            sigma = np.diag((np.ones(10))) * 10 ** 2
            for i in range(maxIter):
                gradient_mu = np.zeros(mu.shape)
                gradient_sigma = np.zeros(sigma.shape)
                theta = [np.zeros(10)] * numSamples
                for j in range(numSamples):
                    theta[j] = np.random.multivariate_normal(mu, sigma)
                    reward[trial_i, i, j] = env.getReward(theta[j])

                assert id(theta[0]) != id(theta[1])
                if use_baseline:
                    b = np.mean(reward[trial_i, i, :])
                else:
                    b = 0

                for j in range(numSamples):
                    derivative_gaussian = log_gauss_derivative(theta[j], mu, sigma)
                    gradient_mu += derivative_gaussian[0] * (reward[trial_i, i, j] - b)
                    if change_sigma:
                        gradient_sigma += derivative_gaussian[1] * (reward[trial_i, i, j] - b)

                mu = mu + alpha_i * gradient_mu / numSamples

                # Learn covariance
                if change_sigma:
                    gradient_sigma = np.diag(np.diag(sigma))  # Only use elements on diagonal
                    sigma = sigma + alpha_i * gradient_sigma / numSamples  # Normalize by number of samples

                    # Set all diagonal elements below lower bound to the lower bound
                    sigma[range(numDim), range(numDim)] = np.maximum(sigma[range(numDim), range(numDim)], sigma_lb)

        reward_mean = np.mean(reward, axis=(0, 2))
        reward_std = np.empty(reward_mean.shape)
        for i in range(maxIter):
            reward_std[i] = np.sqrt(np.sum((reward[:, i, :].ravel() - reward_mean[i]) ** 2) / (numTrials * numSamples))

        ax0.plot(np.arange(maxIter), reward_mean, label=f"Alpha: {alpha_i}")
        plt.fill_between(np.arange(maxIter), reward_mean - 2 * reward_std, reward_mean + 2 * reward_std, alpha=0.5,
                         edgecolor='#1B2ACC', facecolor='#089FFF')

    # ax0.set_title("Subtracted baseline policy gradient - Variance learned")
    ax0.set_xlabel("Number of iterations")
    ax0.set_ylabel("Mean return for all runs")
    ax0.legend()
    ax0.grid()
    plt.tight_layout()

    return fig0


fig = exploration(alpha=(0.1,), use_baseline=False, change_sigma=False)
plt.gca().set_title("Normal policy gradient")
plt.tight_layout()
fig.savefig("Normal policy gradient.pdf")
print("si")

fig = exploration(alpha=(0.1,), use_baseline=True, change_sigma=False)
plt.gca().set_title("Subtracted baseline policy gradient")
plt.tight_layout()
fig.savefig("Subtracted baseline policy gradient.pdf")
print("si")

fig = exploration(alpha=(0.1, 0.2, 0.4), use_baseline=True, change_sigma=False)
plt.gca().set_title("Subtracted baseline policy gradient\nVarying alpha")
plt.tight_layout()
fig.savefig("Subtracted baseline policy gradient_Varying alpha.pdf")
print("si")

fig = exploration(alpha=(0.4,), use_baseline=True, change_sigma=True, sigma_lb=1)
plt.gca().set_title("Subtracted baseline policy gradient\nVariance learned")
plt.tight_layout()
fig.savefig("Subtracted baseline policy gradient_Variance learned.pdf")