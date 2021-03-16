import numpy as np
import matplotlib.pyplot as plt

def get_data(PATH):
    data = np.loadtxt(PATH)
    return data[0, :], data[1, :]

X_train, y_train = get_data("data_ml/training_data.txt")
X_val, y_val = get_data("data_ml/validation_data.txt")

""" <<< ----- Kernel function ----->>>> """
def phi_function(x, degree):
    x_poly = []
    for i in range(degree):
        x_poly.append(np.sin(2**i * x))
    return np.asarray(x_poly).T


def kernel(xi, xj, sigma=0.15):
    return np.exp(- abs(xi - xj) ** 2 / sigma ** 2)

def exponential_squared_kernel(X, sigma):
    K = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            K[i, j] = kernel(X[i], X[j], sigma)
    return K

def kernel_regression(x, X, y, K, sigma):
    f_x = np.zeros((len(x)))
    for i in range(len(x)):
        k = kernel(x[i], X, sigma)
        f_x[i] = k.T @ np.linalg.inv(K) @ y
    return f_x

def linear_regression(x, y, d):
    X = phi_function(x, d)
    return np.linalg.inv(X.T @ X) @ X.T @ y


def predict_y(X, w_hat, d=0):
    return phi_function(X, d) @ w_hat


def mean_square_error(target, prediction):
    return np.sqrt(((target - prediction) ** 2).mean())


def plot_points(X, y, title, color):
    plt.scatter(X, y, label=title, c=color)


def plot_line(y, degree, color):
    x = np.linspace(0, 6, len(y))
    plt.plot(x, y, c=color, label="Line with {} features".format(degree))
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")


def leave_one_out(X, y, index):
    return np.delete(X, index), np.delete(y, index), X[index], y[index]


def compute_mean(errors):
    return errors.mean(axis=1)


def compute_variance(errors):
    return errors.std(axis=1) ** 2


def task1c():
    x = np.arange(0, 6.01, 0.01)

    degree = [2, 3, 9]
    colors = ['r', 'g', 'b']
    plt.figure()
    plot_points(X_train, y_train, "Train Data", 'black')
    plot_points(X_val, y_val, "Val Data", "orange")
    for d, c in zip(degree, colors):
        w_hat = linear_regression(X_train, y_train, d)
        y_pred = predict_y(x, w_hat, d)
        plot_line(y_pred, d, c)

    plt.legend()
    plt.show()


def task1d_e():
    degree = np.arange(1, 9 + 1)
    errors = np.zeros((len(degree), 2))

    for i, d in enumerate(degree):
        """ <<<----- Learning process ---->>>> """
        w_hat = linear_regression(X_train, y_train, d)
        y_train_pred = predict_y(X_train, w_hat, d)
        errors[i, 0] = mean_square_error(y_train, y_train_pred)

        """ <<<----- validation process ---->>>> """
        y_val_pred = predict_y(X_val, w_hat, d)
        errors[i, 1] = mean_square_error(y_val, y_val_pred)

    plt.figure()
    plt.title("Root Mean Square Error (RMSE)")
    plt.bar(degree - 0.2, errors[:, 0], color='b', width=0.4, label='train data')
    plt.bar(degree + 0.2, errors[:, 1], color='r', width=0.4, label='val data')
    plt.xlabel("features")
    plt.ylabel("error")
    plt.legend()
    plt.show()


def task1f():
    degree = np.arange(1, 20 + 1)
    errors = np.zeros((len(degree), len(y_train) - 1))
    for i, d in enumerate(degree):
        for j in range(len(y_train) - 1):
            xtrain, ytrain, xtest, ytest = leave_one_out(X_train, y_train, j)

            """ <<<----- Learning process ---->>>> """
            w_hat = linear_regression(xtrain, ytrain, d)
            y_train_pred = predict_y(xtest, w_hat, d)
            errors[i, j] = mean_square_error(ytest, y_train_pred)

    mean = compute_mean(errors)
    variance = compute_variance(errors)
    print(mean, variance)
    print("minimum mean-error: {}, idx: {}; mimumum variance-variance: {}, idx: {};".format(min(mean),
                                                                                            np.argmin(mean) + 1,
                                                                                            min(variance),
                                                                                            np.argmin(variance) + 1))

    plt.figure()
    plt.title("Mean/Variance of Root Mean Square Error (RMSE)")

    plt.errorbar(degree, mean, variance, linestyle='None', marker='_')

    #plt.bar(degree - 0.2, mean, color='b', width=0.4, label='mean')
    #plt.bar(degree + 0.2, variance, color='r', width=0.4, label='variance')
    plt.xlabel("features")
    plt.ylabel("error")
    plt.legend()
    plt.show()

def task1h():
    sigma = 0.15
    n = 3
    K = exponential_squared_kernel(X_train, sigma)
    errors = np.zeros((1, 2))

    f_x_train = kernel_regression(X_train, X_train, y_train, K, sigma)
    errors[0, 0] = mean_square_error(y_train, f_x_train)

    """ <<<----- validation process ---->>>> """
    f_x_val = kernel_regression(X_val, X_train, y_train, K, sigma)
    errors[0, 1] = mean_square_error(y_val, f_x_val)

    print(errors)

    x = np.arange(0, 6.01, 0.01)
    f_x = kernel_regression(x, X_train, y_train, K, sigma)
    plt.figure()
    plt.title("Kernel Regression")
    plot_points(X_train, y_train, "Train Data", 'black')
    plot_points(X_val, y_val, "Val Data", "orange")
    plt.plot(x, f_x, c='r', label="Line")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.legend()
    plt.show()


def main():
    #task1c()
    #task1d_e()
    task1f()
    #task1h()


if __name__ == '__main__':
    main()