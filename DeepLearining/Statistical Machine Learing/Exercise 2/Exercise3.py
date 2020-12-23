import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


def computeHistograms(data, size, axs, label):
    bin_edges = np.arange(min(data), max(data) + size, size)
    axs.hist(data, bins=bin_edges, density=1, color='blue',edgecolor='black', label=label)
    axs.legend(loc='upper left', frameon=False)

def probability_density_estimate(h, data, X):
    prob_density = []
    for x in X:
        kernel = gaussian_kernel(h, data - x) / (len(data) * h)
        prob_density.append(kernel)
    return np.asarray(prob_density)

def kernel(h, data, X):
    prob_density_gassian = []
    prob_density_parzen_window = []
    prob_desnity_epanechnikov_kernel = []
    for x in X:
        xi = data-x
        k = gaussian_kernel(h, xi) / (len(data) * h)
        prob_density_gassian.append(k)

        k = parzen_window(h, xi)
        prob_density_parzen_window.append(k)

        k = epanechnikov_kernel(h, xi)
        prob_desnity_epanechnikov_kernel.append(k)

    return prob_density_gassian, prob_density_parzen_window, prob_desnity_epanechnikov_kernel

def gaussian_kernel(h, x):
    return np.sum(np.exp(- x**2 / (2 * h**2))) / np.sqrt(2 * np.pi * h**2)

def parzen_window(h, u):
    parzen = []
    for x in u:
        if(x <= h/2):
            parzen.append(1)
        else:
            parzen.append(0)
    return np.sum(np.asarray(parzen))

def epanechnikov_kernel(h, u):
    kernel = []
    for x in u:
        kernel.append(max(0, 0.75*(1 - x/h)**2))
    return np.sum(np.asarray(kernel))

def log_likelihood(probability):
    return np.sum(np.log(probability))

def main():
    nonParamTrain = np.genfromtxt('dataSets/nonParamTrain.txt')
    nonParamTest = np.genfromtxt('dataSets/nonParamTest.txt')

    size_bins = [0.02, 0.5, 2.0]
    for index, size in enumerate(size_bins):
        fig, axs = plt.subplots(1, 2)
        computeHistograms(nonParamTrain, size, axs[0], 'Train_data')
        computeHistograms(nonParamTest, size, axs[1], 'Test_data')
        plt.title('Histogramm with ' + str(size))
        plt.xlabel("Data")
        plt.ylabel("Counts")
        plt.show()

    g_kernel = [0.03, 0.2, 0.8]
    for sig in g_kernel:
        density_train = probability_density_estimate(sig, nonParamTrain, nonParamTrain)
        density_test = probability_density_estimate(sig, nonParamTest, nonParamTest)

        log_likili_train = log_likelihood(density_train)
        log_likeli_test = log_likelihood(density_test)
        print('Log-Likelihood (Train/Test) for {}: {}, {}'.format(sig, log_likili_train, log_likeli_test))

        plt.scatter(nonParamTrain, density_train, label='Train', color='r')
        plt.scatter(nonParamTest, density_test, label='Test', color='b')
        plt.title("probability density estimate using a Gaussian kernel " +  str(sig))
        plt.xlabel('x')
        plt.ylabel('p(x)')
        plt.xlim(-4, 8)
        plt.legend()
        plt.show()

    for sig in g_kernel:
        gaussian_train, parzen_train, epanechnikov_train = kernel(sig, nonParamTrain, nonParamTrain)
        gaussian_test, parzen_test, epanechnikov_test = kernel(sig, nonParamTest, nonParamTest)
        print('Log-Likelihood Train (gaussian, parzen, epanechnikov) for {}: {}, {}, {}'.format(sig, log_likelihood(gaussian_train),
                                                                                            log_likelihood(parzen_train),
                                                                                            log_likelihood(epanechnikov_train)))
        print('Log-Likelihood Test (gaussian, parzen, epanechnikov) for {}: {}, {}, {}'.format(sig, log_likelihood(gaussian_test),
                                                                                            log_likelihood(parzen_test),
                                                                                            log_likelihood(epanechnikov_test)))


if __name__ == '__main__':
    main()