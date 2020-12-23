import numpy as np
import matplotlib.pyplot as plt

#Root Mean Squared Error
def RMSD(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def show_data(x, y, name):
    plt.scatter(x, y, color='k', label=name, s=10)
    plt.legend()
    plt.xlabel('input')
    plt.ylabel('output')

def linear_regression(X, y, alpha, d = 1):
    X = polynomial_matrix(X, d)
    n, m = X.shape
    I = np.identity(m)
    return np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y

def compute_prediction(X, w_hat, d=1):
    return polynomial_matrix(X, d) @ w_hat

def polynomial_matrix(X, d):
    x = []
    for i in range(0, d + 1):
        x.append(np.power(X, i))
    return np.asarray(x).T

def cross_validation(X_data, y_data, i, size=0.2):
    train_size = len(y_data) * size
    X_val, y_test, X_train, y_val = [], [], [], []

    min = round(train_size * i)
    max = round(train_size * i + train_size)
    for idx, (X, y) in enumerate(zip(X_data, y_data)):
       if(idx >= min and idx < max):
           X_val.append(X)
           y_test.append(y)
       else:
           X_train.append(X)
           y_val.append(y)

    return np.asarray(X_train), np.asarray(X_val), np.asarray(y_val), np.asarray(y_test)

def log_likelihood(p):
    return np.sum(np.log(p)) / len(p)

def likelihood(mu, sigma, x):
    factor = 1 / np.sqrt(2*np.pi*sigma**2)
    return factor * np.exp(-0.5 * ((x - mu)/ sigma)**2)


def baysian_linear_regression(phi, y, alpha, beta):
    print(alpha, beta, " alpha und beta")
    n, m = phi.shape
    I = np.identity(m)

    mean_matrix = np.linalg.inv((alpha / beta) * I + phi.T @ phi) @ phi.T @ y
    variance_matrix = np.linalg.inv(alpha*I + beta * phi.T @ phi)

    def predictive_mean(x):
        return x.T @ mean_matrix

    def predictive_variance(x):
        return 1/beta + x.T @ variance_matrix @ x

    means = []
    variance = []
    likeli = []
    for x in phi:
        m = predictive_mean(x)
        means.append(m)
        v = predictive_variance(x)
        variance.append(v)
        likeli.append(likelihood(m, np.sqrt(v), x))
    std = np.asarray(np.sqrt(variance))
    means = np.asarray(means)

    print("Average Error: ", RMSD(means, y))
    print("average Likelihood: ",  log_likelihood(np.asarray(likeli)[:, 1]))

    c = [1, 2, 3]
    X = np.arange(-1, 1, 0.01)
    X = np.c_[np.ones(len(X)), X]

    means = []
    variances = []
    for n in X:
        means.append(predictive_mean(n))
        variances.append(predictive_variance(n))
    means = np.asarray(means)
    std = np.sqrt(np.asarray(variances))

    fig, axs = plt.subplots(len(c))
    for idx, k in enumerate(c):
        axs[idx].plot(X[:, 1], means, label='mean_line', color='r')
        axs[idx].fill_between(X[:, 1], (means - k * std), (means + k * std))
        axs[idx].scatter(phi[:, 1], y, label='datapoints', c='b')
        axs[idx].set_title('k = {}'.format(k))
        axs[idx].legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
    plt.show()

def baysian_regression(phi, phi_test, X_data, y, alpha, beta):
    n, m = phi.shape
    I = np.identity(m)
    mean_matrix = np.linalg.inv((alpha / beta) * I + phi.T @ phi) @ phi.T @ y
    variance_matrix = np.linalg.inv(alpha * I + beta * phi.T @ phi)

    def predictive_mean(x):
        return x.T @ mean_matrix

    def predictive_variance(x):
        return 1 / beta + x.T @ variance_matrix @ x

    means = []
    variances = []
    likeli = []
    for row in phi:
        m = predictive_mean(row)
        means.append(m)
        v = predictive_variance(row)
        variances.append(v)
        likeli.append(likelihood(m, np.sqrt(v), row))

    means = []
    variances = []
    likeli = []
    for row in phi_test:
        m = predictive_mean(row)
        means.append(m)
        v = predictive_variance(row)
        variances.append(v)
        likeli.append(likelihood(m, np.sqrt(v), row))


    means = np.asarray(means)
    print("1e) Average Error (Train): ", RMSD(means, y))
    print("1e) average Likelihood (Train): ",  log_likelihood(np.asarray(likeli)[:, 1]))


    print("1e) Average Error (Test): ", RMSD(means, y))
    print("1e) average Likelihood (Test): ",  log_likelihood(np.asarray(likeli)[:, 1]))

    c = [1, 2, 3]
    X_val = np.linspace(-1, 1, len(y))
    X_val = squad_exponentional_matrix(20, X_val)

    means = []
    variances = []
    for x in X_val:
        means.append(predictive_mean(x))
        variances.append(predictive_variance(x))

    means = np.asarray(means)
    std = np.sqrt(np.asarray(variances))

    X = np.linspace(-1, 1, len(y))

    fig, axs = plt.subplots(len(c))
    for idx, k in enumerate(c):
        axs[idx].plot(X, means, label='mean_line', color='r')
        axs[idx].fill_between(X, (means - k * std), (means + k * std))
        axs[idx].scatter(X_data, y, label='datapoints', c='b')
        axs[idx].set_title('k = {}'.format(k))
        axs[idx].legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
    plt.show()

def squad_exponentional_matrix(k, X, beta = 10):
    phi = []
    for i in range(k):
        alpha = np.ones(len(X)) * i * 0.1 - 1
        vec = np.exp(-0.5 * beta * (X - alpha) ** 2)
        phi.append(vec)
    return np.asarray(phi).T

def doBayesianRegression(phi, X_data, y, alpha, beta, sigma = 0.1):
    n, m = phi.shape
    I = np.identity(m)

    mu = sigma**-2 * np.linalg.inv((alpha/beta)*I) @ phi.T @ y
    sig = sigma**-2 * phi.T @ phi + (alpha / beta) * I

    def logMarginalLikelihood(x):
        return ((phi.shape[1] + 1) / 2) * np.log(alpha/beta) \
               - (phi.shape[0] / 2) * np.log(sig ** 2) \
               - 0.5 * np.linalg.norm(y - x @ mu)**2 / sig**2 \
               + 0.5 * (alpha/beta) * mu.T @ mu \
               - 0.5 * np.log(sig) \
               - 0.5 * phi.shape[0]*np.log(2*np.pi)


def calc_error_of_train(phi_train, phi_test, y_train, y_test, alpha, beta):
    n, m = phi_train.shape
    I = np.identity(m)

    mean_matrix = np.linalg.inv((alpha / beta) * I + phi_train.T @ phi_train) @ phi_train.T @ y_train
    variance_matrix = np.linalg.inv(alpha * I + beta * phi_train.T @ phi_train)

    def predictive_mean(x):
        return x.T @ mean_matrix

    def predictive_variance(x):
        return 1 / beta + x.T @ variance_matrix @ x

    means = []
    variance = []
    likeli = []
    for x in phi_test:
        m = predictive_mean(x)
        means.append(m)
        v = predictive_variance(x)
        variance.append(v)
        likeli.append(likelihood(m, np.sqrt(v), x))
    std = np.asarray(np.sqrt(variance))
    means = np.asarray(means)
    print("1e) Average Error: ", RMSD(means, y_test))
    print("1e) average Likelihood: ", log_likelihood(np.asarray(likeli)[:, 1]))


def main():
    #inizialization
    ridge_coefficient = 0.01
    train_data = np.genfromtxt('dataSets/lin_reg_train.txt', delimiter=' ')
    test_data = np.loadtxt('dataSets/lin_reg_test.txt', delimiter=' ')

    X_train = train_data[:, 0] #np.c_[np.ones(len(train_data[:, 0])), train_data[:, 0]]
    y_train = train_data[:, 1]

    X_test = test_data[:, 0] #np.c_[np.ones(len(test_data[:, 0])), test_data[:, 0]]
    y_test = test_data[:, 1]

    """ 1a) """
    w_hat = linear_regression(X_train, y_train, ridge_coefficient)
    y_pred_train = compute_prediction(X_train, w_hat)
    e_train = RMSD(y_pred_train, y_train)

    y_pred_test = compute_prediction(X_test, w_hat)
    e_test = RMSD(y_pred_test, y_test)

    print("RMSD Train {}, \t RMSD Test {}".format(e_train, e_test))

    plt.figure()
    plt.plot(X_train, y_pred_train, c='b', label='Regression Line')
    show_data(X_train, y_train, 'train_data')
    plt.show()

    """ 1b) """
    degrees = [2, 3, 4, 20]
    color_map = ['b', 'b', 'b']

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.figure(figsize=(10, 10))

    i = 221
    show_data(X_train, y_train, 'train_data')
    for d, color in zip(degrees, color_map):
        w_hat = linear_regression(X_train, y_train, ridge_coefficient, d)
        y_pred = compute_prediction(X_train, w_hat, d)
        error = RMSD(y_pred, y_train)
        print('1b) Error train: ', error)

        y_pred_test = compute_prediction(X_test, w_hat, d)
        error = RMSD(y_pred_test, y_test)
        print("1b) error test: ", error)

        #sorted_zip = sorted(zip(X_train, y_pred))
        #x, y = zip(*sorted_zip)
        #plt.subplot(i)
        #i = i + 1
        #show_data(X_train, y_train, 'train_data')
        #plt.plot(x, y, c=color, label='Regression Line dim=' + str(d))
        x = np.linspace(-1, 1, 50)
        y_pred = compute_prediction(x, w_hat, d)
        plt.plot(x, y_pred, c=color, label='Regression Line dim=' + str(d))
        plt.legend()
    plt.title(label='Polynomial Regression: ')
    plt.show()

    error_2, error_3, error_4 = [], [], []
    """ 1c) """
    for i in range(5):
        X_train, X_val, y_train, y_val = cross_validation(train_data[:, 0], train_data[:, 1], i)
        for d in degrees:
            w_hat = linear_regression(X_train, y_train, ridge_coefficient, d)
            y_pred = compute_prediction(X_train, w_hat, d)
            error_train = RMSD(y_pred, y_train)

            y_pred = compute_prediction(X_val, w_hat, d)
            error_val = RMSD(y_pred, y_val)

            y_pred = compute_prediction(X_test, w_hat, d)
            error_test = RMSD(y_pred, y_test)

            if(d == 2):
                error_2.append([error_train, error_val, error_test])
            elif(d == 3):
                error_3.append([error_train, error_val, error_test])
            else:
                error_4.append([error_train, error_val, error_test])

            #print("Dim: {}, i: {} -->  Error Train: {}, Error Val: {}, Error Test: {}".format(d, i, error_train, error_val, error_test))
#
    #Calculate Average of
    error_2 = np.asarray(error_2)
    error_3 = np.asarray(error_3)
    error_4 = np.asarray(error_4)
    set = ['Train', 'Val', 'Test']
    for i, set in zip(range(error_2.shape[1]), set):
        print("Average value of Error: {} -->  dim2: {}, dim3: {}, dim4: {}".format(set, error_2[:, i].mean(),
                                                                                   error_3[:, i].mean(), error_4[:, i].mean()))

    """ 1d) """
    ridge_coefficient = 0.01
    X_train = train_data[:, 0]
    y_train = train_data[:, 1]
    X_test = test_data[:, 0]
    y_test = test_data[:, 1]
    baysian_linear_regression(np.c_[np.ones(len(X_train)), X_train], y_train, ridge_coefficient, 1/0.01)
    baysian_linear_regression(np.c_[np.ones(len(X_test)), X_test], y_test, ridge_coefficient, 1/0.01)
    calc_error_of_train(np.c_[np.ones(len(X_train)), X_train], np.c_[np.ones(len(X_test)), X_test], y_train, y_test, ridge_coefficient, 1 / 0.01)

    """ 1e) """
    phi_train = squad_exponentional_matrix(20, X_train)
    phi_test = squad_exponentional_matrix(20, X_test)
    #calc_error_of_train(phi_train, phi_test, y_train, y_test, ridge_coefficient, 1/0.01)
    baysian_regression(phi_train, phi_test, X_train, y_train, ridge_coefficient, 1/0.01)

    """ 1f) """
    #beta = [1, 10, 100]
    #for b in beta:
    #    baysian_regression(phi_train, phi_test, X_train, y_train, 0.01, b)
    #    calc_error_of_train(phi_train, phi_test, y_train, y_test, ridge_coefficient, b)

if __name__ == '__main__':
    main()