import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

""" 2c) Biased ML Estimate """
class Data:
    mean = 0

    variance_unbiased = 0
    variance_biased = 0

    cov_biased = 0
    cov_unbiased = 0

    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name

        self.calc_mean()
        self.calc_sample_variance_biased_and_unbiased()
        self.covariance_setup()

    def calc_covariance_biased(self, x, y):
        if(x == 0 and y == 0):
            return self.variance_biased[x]
        elif(x == 1 and y == 1):
            return self.variance_biased[y]
        else:
            sum = 0
            for i in range(self.x.shape[0]):
                sum += (self.x[i] - self.mean[0])*(self.y[i] - self.mean[1])
            return sum / self.x.shape[0]


    def calc_covariance_unbiased(self, x, y):
        if (x == 0 and y == 0):
            return self.variance_unbiased[x]
        elif (x == 1 and y == 1):
            return self.variance_unbiased[y]
        else:
            sum = 0
            for i in range(self.x.shape[0]):
                sum += (self.x[i] - self.mean[0])*(self.y[i] - self.mean[1])
            return sum / (self.x.shape[0] - 1)

    def covariance_setup(self):
        self.cov_biased = np.asarray([self.calc_covariance_biased(x, y) for x in range(2) for y in range(2)]).reshape(2, 2)
        self.cov_unbiased = np.asarray([self.calc_covariance_unbiased(x, y) for x in range(2) for y in range(2)]).reshape(2, 2)

    def calc_mean(self):
        val_x = 0
        val_y = 0
        for i in range(self.x.shape[0]):
            val_x += self.x[i]
            val_y += self.y[i]
        self.mean = np.asarray([val_x / len(self.x), val_y / len(self.y)])

    def calc_sample_variance_biased_and_unbiased(self):
        val_x = 0
        val_y = 0
        for i in range(self.x.shape[0]):
            val_x += (self.x[i] - self.mean[0])**2
            val_y += (self.y[i] - self.mean[1])**2

        self.variance_unbiased = np.asarray([val_x / (len(self.x) - 1), val_y / (len(self.y) - 1)])
        self.variance_biased = np.asarray([val_x / len(self.x), val_y / len(self.y)])
""" end 2c) """


def get_data(name :str) -> np.asarray:
    file = open(name, 'r')
    x = []
    y = []
    for i in file.readlines():
        line = i.split()
        x.append(float(line[0]))
        y.append(float(line[1]))
    file.close()
    return np.asarray(x), np.asarray(y)

def show_data(densEst1, densEst2):
    plt.figure()
    plt.plot(densEst2.x, densEst2.y, 'bo', marker='x', label='densEst2')
    plt.plot(densEst1.x, densEst1.y, 'ro', marker='o', label='densEst1')
    plt.title('Vizualization of DensEst1 and DensEst2')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def calc_multi_gaussian(x, cov, mean):
    factor = 1 / np.sqrt((2*np.pi)**2 * np.linalg.det(cov))
    z = []
    for pos in x:
        multi = factor * np.exp(-0.5 * (pos - mean) @ np.linalg.inv(cov) @ (pos.T - mean.T))
        z.append(multi)
    return np.asarray(z)

def multi_gaussian(data):
    z = calc_multi_gaussian(np.c_[data.x, data.y], data.cov_unbiased, data.mean)
    fig, ax2 = plt.subplots()
    ax2.tricontour(data.x, data.y, z, levels=14, linewidths=0.5, colors='k')
    cntr2 = ax2.tricontourf(data.x, data.y, z, levels=14, cmap="RdBu_r")

    fig.colorbar(cntr2, ax=ax2)
    ax2.plot(data.x, data.y, 'ko', ms=3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Contour plot of multivariate gaussian: ' +  data.name)
    plt.subplots_adjust(hspace=0.5)
    plt.show()

def prior_probability(densEst1, densEst2):
    prior_dens1 = len(densEst1.x) / (len(densEst1.x) + len(densEst2.x))
    prior_dens2 = len(densEst2.x) / (len(densEst1.x) + len(densEst2.x))
    return np.asarray([prior_dens1, prior_dens2])

def calculate_posterori(densEst1, densEst2, xx, yy):
    def likelihood(xx, yy, cov, mean):
        likeli = np.zeros(xx.shape)
        for i in range(xx.shape[1]):
            likeli[:, i] = calc_multi_gaussian(np.c_[xx[:, i], yy[:, i]], cov, mean)
        return likeli

    prior_dens1, prior_dens2 = prior_probability(densEst1, densEst2)

    likelihood_dens1 = likelihood(xx, yy, densEst1.cov_unbiased, densEst1.mean)
    likelihood_dens2 = likelihood(xx, yy, densEst2.cov_unbiased, densEst2.mean)
    normalization = likelihood_dens1 * prior_dens1 + likelihood_dens2 * prior_dens2

    posterori_dens1 = likelihood_dens1 * prior_dens1 / normalization
    posterori_dens2 = likelihood_dens2 * prior_dens2 / normalization


    return np.asarray(posterori_dens1), np.asarray(posterori_dens2)

def decision(Z1, Z2):
    Z = np.zeros(Z1.shape)
    for i in range(Z1.shape[0]):
        for j in range(Z1.shape[1]):
            if(Z1[i, j] >= Z2[i, j]):
                Z[i, j] = 0
            else:
                Z[i, j] = 1
    return Z


def multi_single_plot(densEst1, densEst2):
    cmap_light = ListedColormap(["#FFAAAA", "#AAAAFF"])
    X = np.c_[densEst1.x, densEst1.y]
    x_min1, x_max1 = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min1, y_max1 = X[:, 1].min() - 1, X[:, 1].max() + 1

    X = np.c_[densEst2.x, densEst2.y]
    x_min2, x_max2 = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min2, y_max2 = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min1, x_max1, 0.05), np.arange(y_min1, y_max1, 0.05))
    Z, _ = calculate_posterori(densEst1, densEst2, xx, yy)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap='viridis')
    plt.colorbar(label='probability')
    plt.title('posterori probability of ' + densEst1.name)
    # Plot also the training points
    plt.scatter(densEst1.x, densEst1.y, marker='x', color='red', label=densEst1.name)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.legend()
    plt.show()


    xx, yy = np.meshgrid(np.arange(x_min2, x_max2, 0.05), np.arange(y_min2, y_max2, 0.05))
    _, Z = calculate_posterori(densEst1, densEst2, xx, yy)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap='viridis')
    plt.colorbar(label='probability')
    plt.title('posterori probability of ' + densEst2.name)
    # Plot also the training points
    plt.scatter(densEst2.x, densEst2.y, marker='x', color='red', label=densEst2.name)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.legend()
    plt.show()

    """Plot for decision Boundary """
    x_min = min(x_min1, x_min2)     #find min element (x_value) of densEst1 and densEst2
    x_max = max(x_max1, x_max2)     #find max element (x_value) " " " "
    y_min = min(y_min1, y_max1)
    y_max = max(y_max1, y_max2)

    """ Create meshgrid of x and y axis"""
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
    Z1, Z2 = calculate_posterori(densEst1, densEst2, xx, yy)
    Z = decision(Z1, Z2)        #make a decision of the probability

    """ create decision boundary with both dataset in one plot"""
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.title('p(x|C): red: decision for densEst1, Blue: decision for densEst2')
    # Plot also the training points
    plt.scatter(densEst1.x, densEst1.y, marker='x', color='red', label='densEst1')
    plt.scatter(densEst2.x, densEst2.y, marker='x', color='blue', label='densEst2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.legend()
    plt.show()

def main():
    #inizialization
    x_densEst1, y_densEst1 = get_data('dataSets/densEst1.txt')
    x_densEst2, y_densEst2 = get_data('dataSets/densEst2.txt')

    densEst1 = Data(x_densEst1, y_densEst1, 'densEst1')
    densEst2 = Data(x_densEst2, y_densEst2, 'densEst2')

    """ 2d) Class Density """
    show_data(densEst1, densEst2)
    multi_gaussian(densEst1)
    multi_gaussian(densEst2)
    
    """ 2e) Posterior """
    multi_single_plot(densEst1, densEst2)

if __name__ == '__main__':
    main()

